import argparse
import json
import re

parser = argparse.ArgumentParser(description='Merge video clips')
parser.add_argument('--files_to_merge', help='JSON file to merge', required=True)
parser.add_argument('--output_file', help='Output file path for the merged JSON', required=True)
args = parser.parse_args()

def parse_box_from_raw_text(text, coords_pattern=r"{<(\d+)><(\d+)><(\d+)><(\d+)>}"):
    try:
        raw_coords = re.findall(coords_pattern, text)
        if len(raw_coords) < 1:
            raw_coords = re.findall(r"\[([\d\s,]+)\]", text)
            coords = [[float(coord) for coord in xyxy_str.replace(" ", "").split(",")][:4] for xyxy_str in raw_coords]
            coords = []
            for xyxy_str in raw_coords:
                box = []
                for coord in xyxy_str.replace(" ", "").split(","):
                    box.append(float(coord))
                box = box[:4]
                if len(box) < 4:
                    box = coords[-1]
                    if len(box) < 4:
                        box = [0,0,0,0]
                coords.append(box)
        else:
            coords = [[float(coord) for coord in xyxy_str][:4] for xyxy_str in raw_coords]
        return coords
    except Exception as e:
        print(e)
        return []

def b2str(b):
    return "[" + ",".join([str(int(pos)) for pos in b]) + "]"

results = {}
jsonfile = args.files_to_merge
print(jsonfile)
f = open(jsonfile, "r").read()
# flat_outputs = json.loads(f.read())
lines = [l + "}" for l in f.split("}\n")]
start_frame_idx = 0
is_first_line = True
for line in lines:
    try:
        line = json.loads(line)
    except:
        if len(line) < 4:
            continue
        # print(line)
    _id = line["id"]
    seq_id, clip_id = _id.split("|")
    seq_id = seq_id
    if seq_id not in results: 
        is_first_line = True
        start_frame_idx = 0
    else:
        is_first_line = False
    clip_id = int(clip_id)
    image_size = line["image_size"]
    predict = line["predict"]
    gt = line["gt"]
    pred_bb = parse_box_from_raw_text(predict)
    anno_bb = parse_box_from_raw_text(gt)
    if len(pred_bb) > len(anno_bb):
        pred_bb = pred_bb[:len(anno_bb)]
    elif len(pred_bb) < len(anno_bb):
        pad_len = len(anno_bb) - len(pred_bb)
        for i in range(pad_len):
            pred_bb.append([0., 0., 0., 0.])
    clip_frame_count = len(pred_bb) if is_first_line else len(pred_bb) - 1

    pred_bb = pred_bb[-clip_frame_count:]
    anno_bb = anno_bb[-clip_frame_count:]

    if seq_id not in results: 
        results[seq_id] = {}
        results[seq_id]["predict"] = ",".join(f"Frame {start_frame_idx + i + 1}: {b2str(pred_bb[i])}" for i in range(0, clip_frame_count))
        results[seq_id]["gt"] =  ",".join(f"Frame {start_frame_idx + i + 1}: {b2str(anno_bb[i])}" for i in range(0, clip_frame_count))
        results[seq_id]["image_size"] = image_size
        results[seq_id]["source"] = "unknown"
        results[seq_id]["vid"] = seq_id
        results[seq_id]["id"] = seq_id
        clip_count = 1
    else:
        clip_count += 1
        if clip_count < 1000000000:  # see if metrics go lower when processing longer frames, set to a large value to evaluate the whole video
            results[seq_id]["predict"] += ",".join(f"Frame {start_frame_idx + i + 1}: {b2str(pred_bb[i])}" for i in range(0, clip_frame_count))
            results[seq_id]["gt"] += ",".join(f"Frame {start_frame_idx + i + 1}: {b2str(anno_bb[i])}" for i in range(0, clip_frame_count))
    start_frame_idx += clip_frame_count

with open(args.output_file, "w") as f:
    for line in list(results.values()):
        f.write(json.dumps(line, ensure_ascii=False) + '\n')
