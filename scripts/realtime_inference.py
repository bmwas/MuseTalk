import argparse
import os
import gc  # For garbage collection
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import coord_placeholder
from musetalk.utils.blending import get_image_blending, get_image_prepare_material
from musetalk.utils.utils import load_all_model
import shutil

import threading
import queue

import time

# Load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU available .........")
else:
    print("STOP GPU NOT available!!")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avatar_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it? (y/n): ")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  Creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path, self.latents_out_path])
                    self.prepare_material()
                else:
                    # Load coordinate lists
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list = pickle.load(f)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list = pickle.load(f)
            else:
                print("*********************************")
                print(f"  Creating avatar: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path, self.latents_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f"【bbox_shift】 is changed, you need to re-create it! (c/continue): ")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  Creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path, self.latents_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                # Load coordinate lists
                with open(self.coords_path, 'rb') as f:
                    self.coord_list = pickle.load(f)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list = pickle.load(f)

    def prepare_material(self):
        print("Preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"Copying files from {self.video_path}")
            files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(".png")])
            for filename in files:
                shutil.copyfile(
                    os.path.join(self.video_path, filename),
                    os.path.join(self.full_imgs_path, filename)
                )

        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')))

        print("Extracting landmarks and processing images...")
        coord_list = []
        mask_coords_list = []
        os.makedirs(self.latents_out_path, exist_ok=True)

        for idx, img_path in enumerate(tqdm(input_img_list)):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue  # Skip if image is not readable

            # Detect landmarks and get bounding box
            bbox = self.get_landmark_and_bbox_single_image(img)
            if bbox == coord_placeholder:
                print(f"No face detected in image: {img_path}")
                continue

            coord_list.append(bbox)
            x1, y1, x2, y2 = bbox

            # Crop and resize the face region
            crop_frame = img[y1:y2, x1:x2]
            if crop_frame.size == 0:
                print(f"Empty crop for image: {img_path}")
                continue  # Skip if crop is empty

            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

            # Generate latent representation
            latents = vae.get_latents_for_unet(resized_crop_frame)

            # Save latent representation to disk
            latent_path = os.path.join(self.latents_out_path, f'{str(idx).zfill(8)}.pt')
            torch.save(latents.cpu(), latent_path)  # Save to CPU to free GPU memory

            # Prepare mask and save
            mask, crop_box = get_image_prepare_material(img, bbox)
            mask_path = os.path.join(self.mask_out_path, f'{str(idx).zfill(8)}.png')
            cv2.imwrite(mask_path, mask)
            mask_coords_list.append(crop_box)

            # Clear variables to free memory
            del img, crop_frame, resized_crop_frame, latents, mask
            gc.collect()

        # Save coordinate lists to disk
        with open(self.coords_path, 'wb') as f:
            pickle.dump(coord_list, f)
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(mask_coords_list, f)

        print("Data preparation completed.")

    def get_landmark_and_bbox_single_image(self, img):
        # This function detects face landmarks and returns the bounding box
        # Replace this with your actual face detection code
        # For now, let's assume we are using face_recognition library
        coord_placeholder = (0, 0, 0, 0)
        try:
            import face_recognition
        except ImportError:
            print("face_recognition library is not installed.")
            sys.exit(1)

        face_locations = face_recognition.face_locations(img)
        if len(face_locations) == 0:
            return coord_placeholder
        else:
            top, right, bottom, left = face_locations[0]
            x1, y1, x2, y2 = left, top, right, bottom
            return (x1, y1, x2, y2)

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        print(f"Total frames to process: {video_len}")
        # Load coordinate lists
        with open(self.coords_path, 'rb') as f:
            coord_list = pickle.load(f)
        with open(self.mask_coords_path, 'rb') as f:
            mask_coords_list = pickle.load(f)
        while self.idx < video_len:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            idx = self.idx

            # Read the original frame and mask from disk
            frame_path = os.path.join(self.full_imgs_path, f'{str(idx).zfill(8)}.png')
            mask_path = os.path.join(self.mask_out_path, f'{str(idx).zfill(8)}.png')

            ori_frame = cv2.imread(frame_path)
            if ori_frame is None:
                print(f"Frame not found: {frame_path}")
                self.idx += 1
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Mask not found: {mask_path}")
                self.idx += 1
                continue

            # Get bounding box and mask crop box
            bbox = coord_list[idx]
            mask_crop_box = mask_coords_list[idx]

            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception as e:
                print(f"Error resizing frame at index {idx}: {e}")
                self.idx += 1
                continue

            # Combine frame using blending
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            if not skip_save_images:
                output_frame_path = os.path.join(self.avatar_path, 'tmp', f'{str(self.idx).zfill(8)}.png')
                cv2.imwrite(output_frame_path, combine_frame)
            self.idx += 1

    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        os.makedirs(os.path.join(self.avatar_path, 'tmp'), exist_ok=True)
        print("Start inference")
        # Extract audio features
        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print(f"Processing audio: {audio_path} took {(time.time() - start_time) * 1000:.2f} ms")
        # Inference batch by batch
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0

        # Start processing frames in a separate thread
        process_thread = threading.Thread(
            target=self.process_frames,
            args=(res_frame_queue, video_num, skip_save_images)
        )
        process_thread.start()

        # Generate data generator
        latent_paths = [os.path.join(self.latents_out_path, f'{str(i).zfill(8)}.pt') for i in range(video_num)]
        gen = datagen(
            whisper_chunks,
            None,  # Latent tensors are loaded from disk
            self.batch_size,
            latent_paths=latent_paths
        )
        start_time = time.time()

        for i, (whisper_batch, latent_paths_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=device, dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)

            # Load latents from disk
            latent_batch = []
            for latent_path in latent_paths_batch:
                if os.path.exists(latent_path):
                    latents = torch.load(latent_path, map_location=device)
                    latent_batch.append(latents)
                else:
                    print(f"Latent file not found: {latent_path}")
                    # Create a zero tensor with the correct shape
                    latents = torch.zeros((4, 64, 64), dtype=unet.model.dtype, device=device)
                    latent_batch.append(latents)
            if not latent_batch:
                continue  # Skip if no latents loaded

            latent_batch = torch.stack(latent_batch).to(dtype=unet.model.dtype)

            # Perform inference
            pred_latents = unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample

            # Decode latents
            recon = vae.decode_latents(pred_latents)

            # Put reconstructed frames into queue
            for res_frame in recon:
                res_frame_queue.put(res_frame.cpu().numpy())

        # Wait for processing thread to finish
        process_thread.join()

        if skip_save_images:
            print(f'Total processing time of {video_num} frames without saving images = {time.time() - start_time}s')
        else:
            print(f'Total processing time of {video_num} frames including saving images = {time.time() - start_time}s')

        if out_vid_name and not skip_save_images:
            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
            # Command to convert images to video and combine audio
            cmd_img2video = f"ffmpeg -y -r {fps} -i {self.avatar_path}/tmp/%08d.png -i {audio_path} -c:v libx264 -c:a aac -strict experimental -b:a 192k {output_vid}"
            print(cmd_img2video)
            os.system(cmd_img2video)
            shutil.rmtree(os.path.join(self.avatar_path, 'tmp'))
            print(f"Result saved to {output_vid}")

        print("\n")

if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config",
                        type=str,
                        default="configs/inference/realtime.yaml",
                        )
    parser.add_argument("--fps",
                        type=int,
                        default=25,
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        )
    parser.add_argument("--skip_save_images",
                        action="store_true",
                        help="Whether to skip saving images for better generation speed calculation",
                        )

    args = parser.parse_args()

    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        bbox_shift = inference_config[avatar_id]["bbox_shift"]
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=args.batch_size,
            preparation=data_preparation
        )

        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            print("Inferring using:", audio_path)
            avatar.inference(audio_path,
                             audio_num,
                             args.fps,
                             args.skip_save_images)
