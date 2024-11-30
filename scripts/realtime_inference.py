import argparse
import os
import sys
import shutil
import threading
import queue
import time
import copy
import glob
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import json
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending

# Load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    cap.release()


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
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
                response = input(f"{self.avatar_id} exists, Do you want to re-create it? (y/n) ")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  Creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.load_materials()
            else:
                print("*********************************")
                print(f"  Creating avatar: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f"【bbox_shift】 is changed, you need to re-create it! (c/continue) ")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  Creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.load_materials()

    def load_materials(self):
        # Load materials
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                                 key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)

        # Validate and filter materials
        valid_indices = []
        for idx, bbox in enumerate(self.coord_list_cycle):
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid bbox at idx {idx}: {bbox}")
                continue
            valid_indices.append(idx)

        # Filter lists based on valid indices
        self.coord_list_cycle = [self.coord_list_cycle[i] for i in valid_indices]
        self.frame_list_cycle = [self.frame_list_cycle[i] for i in valid_indices]
        self.mask_coords_list_cycle = [self.mask_coords_list_cycle[i] for i in valid_indices]
        self.mask_list_cycle = [self.mask_list_cycle[i] for i in valid_indices]
        self.input_latent_list_cycle = [self.input_latent_list_cycle[i] for i in valid_indices]

    def prepare_material(self):
        print("Preparing data materials ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='.png')
        else:
            print(f"Copying files from {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.lower().endswith(".png")]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        print("Extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        valid_coord_list = []
        valid_frame_list = []
        idx = -1
        # Marker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx += 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox

            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid bbox at idx {idx}: {bbox}")
                continue

            # Ensure coordinates are within image bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(int(x1), width - 1))
            x2 = max(0, min(int(x2), width))
            y1 = max(0, min(int(y1), height - 1))
            y2 = max(0, min(int(y2), height))

            crop_frame = frame[y1:y2, x1:x2]

            # Check if crop_frame is empty
            if crop_frame.size == 0:
                print(f"Warning: empty crop frame at idx {idx}")
                continue

            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
            valid_coord_list.append(bbox)
            valid_frame_list.append(frame)

        # Use only valid frames and coordinates
        self.frame_list_cycle = valid_frame_list + valid_frame_list[::-1]
        self.coord_list_cycle = valid_coord_list + valid_coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            face_box = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self,
                       res_frame_queue,
                       video_len,
                       skip_save_images):
        print(f"Total frames to process: {video_len}")
        while True:
            if self.idx >= video_len:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            idx_in_cycle = self.idx % len(self.coord_list_cycle)
            bbox = self.coord_list_cycle[idx_in_cycle]
            ori_frame = copy.deepcopy(self.frame_list_cycle[idx_in_cycle])
            x1, y1, x2, y2 = bbox

            # Ensure valid dimensions
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                print(f"Invalid dimensions at idx {self.idx}: width={width}, height={height}")
                self.idx += 1
                continue

            try:
                res_frame = res_frame.astype(np.uint8)
                if res_frame.ndim == 2:
                    res_frame = cv2.cvtColor(res_frame, cv2.COLOR_GRAY2BGR)
                res_frame = cv2.resize(res_frame, (width, height))

                mask = self.mask_list_cycle[idx_in_cycle]
                mask_crop_box = self.mask_coords_list_cycle[idx_in_cycle]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            except Exception as e:
                print(f"Error processing frame at idx {self.idx}: {e}")
                self.idx += 1
                continue

            if not skip_save_images:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx += 1

    def inference(self,
                  audio_path,
                  out_vid_name,
                  fps,
                  skip_save_images):
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("Starting inference...")
        ############################################## Extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print(f"Processing audio: {audio_path} took {(time.time() - start_time) * 1000:.2f} ms")
        ############################################## Inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
        process_thread.start()

        gen = datagen(whisper_chunks,
                      self.input_latent_list_cycle,
                      self.batch_size)
        start_time = time.time()

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=torch.float16)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(device=unet.device, dtype=torch.float16)

            pred_latents = unet.model(latent_batch,
                                      timesteps,
                                      encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
            # Free up GPU memory
            del pred_latents
            del recon
            torch.cuda.empty_cache()

        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()

        if skip_save_images:
            print('Total processing time of {} frames without saving images: {:.2f}s'.format(
                video_num,
                time.time() - start_time))
        else:
            print('Total processing time of {} frames including saving images: {:.2f}s'.format(
                video_num,
                time.time() - start_time))

        if out_vid_name is not None and not skip_save_images:
            # Optional
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png " \
                            f"-vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p " \
                            f"-crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"Result is saved to {output_vid}")
        print("\n")


if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection 
    and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes 
    MuseTalk real-time.
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
                        default=1,  # Reduced batch size to prevent CUDA OOM
                        )
    parser.add_argument("--skip_save_images",
                        action="store_true",
                        help="Whether to skip saving images for better generation speed calculation",
                        )

    args = parser.parse_args()

    # Set PyTorch memory allocation config to prevent fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

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
            preparation=data_preparation)

        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            print("Inferring using:", audio_path)
            avatar.inference(audio_path,
                             audio_num,
                             args.fps,
                             args.skip_save_images)
