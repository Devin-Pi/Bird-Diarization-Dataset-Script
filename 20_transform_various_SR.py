import os
import json
import glob
import argparse

# python 20_transform_various_SR.py --dataset_root /workspaces/bird_data/filtered_32k_16k_OVERLAP_50 --output_root /workspaces/filtered_32k_16k_OVERLAP_50_rttm/ --target_srs 32000 16000

# ==========================================
# 1. global setting
# ==========================================
parser = argparse.ArgumentParser(
    description="Generate Kaldi/Pyannote Manifests (SCP, RTTM, UEM)")


parser.add_argument('--dataset_root', type=str, required=True,
                    help='Root directory of the generated dataset')


parser.add_argument('--output_root', type=str, required=True,
                    help='Directory to save the generated manifests')


parser.add_argument('--target_srs', type=int, nargs='+', default=[32000],
                    help='List of target sample rates')

args = parser.parse_args()

DATASET_ROOT = args.dataset_root
OUTPUT_ROOT = args.output_root
TARGET_SRS = args.target_srs

# 子集列表保持固定
SUBSETS = ['train', 'dev', 'test']

# 设置 umask 保证生成的文件权限正常
os.umask(0)

# ==========================================


def generate_manifests_for_sr(subset_name, sr, dataset_root, output_root):

    sr_tag = f"{sr // 1000}k"

    # 1. 确定输入目录结构
    json_src_dir = os.path.join(dataset_root, subset_name, "json")
    wav_folder_name = f"wav{sr_tag}"
    wav_src_dir = os.path.join(dataset_root, subset_name, wav_folder_name)

    # 2. 确定输出目录 -> output_root/32k/train
    out_dir = os.path.join(output_root, sr_tag, subset_name)

    if not os.path.exists(json_src_dir):

        print(f"⚠️ JSON folder not found: {json_src_dir}, skipping.")
        return
    if not os.path.exists(wav_src_dir):
        print(f"⚠️ WAV folder not found: {wav_src_dir}, skipping.")
        return

    os.makedirs(out_dir, mode=0o777, exist_ok=True)

    scp_path = os.path.join(out_dir, "wav.scp")
    rttm_path = os.path.join(out_dir, "ref.rttm")
    uem_path = os.path.join(out_dir, "all.uem")

    json_files = sorted(glob.glob(os.path.join(json_src_dir, "*.json")))

    if not json_files:
        print(f"⚠️ No JSON files found in {json_src_dir}.")
        return

    print(f"🚀 Processing [{subset_name}] @ Folder [{wav_folder_name}] ...")

    processed_count = 0

    with open(scp_path, 'w', encoding='utf-8') as f_scp, \
            open(rttm_path, 'w', encoding='utf-8') as f_rttm, \
            open(uem_path, 'w', encoding='utf-8') as f_uem:

        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"   ❌ Failed to read {json_path}: {e}")
                continue

            # file name
            json_filename = os.path.basename(json_path)
            base_name = os.path.splitext(json_filename)[0]

            duration = data['duration_sec']

            # wav file name
            wav_filename = f"{base_name}.wav"
            wav_abs_path = os.path.join(wav_src_dir, wav_filename)

            if not os.path.exists(wav_abs_path):
                print(f"   ⚠️ Wav file missing: {wav_abs_path}")
                continue

            # recording ID
            rec_id = f"{base_name}_{sr_tag}".replace('.', '_')

            # wav.scp ---
            f_scp.write(f"{rec_id} {wav_abs_path}\n")

            # all.uem ---
            f_uem.write(f"{rec_id} 1 0.0 {duration:.4f}\n")

            #  ref.rttm ---
            events = data.get('events', [])
            events.sort(key=lambda x: x['start'])

            for evt in events:
                start = evt['start']
                end = evt['end']
                dur = end - start
                label = evt['label']

                if dur > 0.001:
                    line = (
                        f"SPEAKER {rec_id} 1 "
                        f"{start:.4f} {dur:.4f} "
                        f"<NA> <NA> {label} <NA>\n"
                    )
                    f_rttm.write(line)

            processed_count += 1

    try:
        os.chmod(scp_path, 0o666)
        os.chmod(rttm_path, 0o666)
        os.chmod(uem_path, 0o666)
    except:
        pass

    print(f"✅ {subset_name} @ {sr_tag} done! Generated {processed_count} entries.")


def main():
    print("\n" + "="*60)
    print(f"📂 Dataset Root: {DATASET_ROOT}")
    print(f"🎯 Output Root:  {OUTPUT_ROOT}")
    print(f"🎛️ Target SRs:   {TARGET_SRS}")
    print("="*60)

    for sr in TARGET_SRS:
        for subset in SUBSETS:
            generate_manifests_for_sr(subset, sr, DATASET_ROOT, OUTPUT_ROOT)

    print("="*60)
    print("🎉 All manifests generated successfully.\n")


if __name__ == "__main__":
    main()
