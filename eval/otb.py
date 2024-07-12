import json
import re
from argparse import ArgumentParser

import torch


def calc_err_center(pred_bb, anno_bb, normalized=False):
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
    return err_center


def calc_iou_overlap(pred_bb, anno_bb):
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union


def calc_seq_err_robust(pred_bb, anno_bb, dataset, target_visible=None):
    pred_bb = pred_bb.clone()

    # Check if invalid values are present
    if torch.isnan(pred_bb).any() or (pred_bb[:, 2:] < 0.0).any():
        raise Exception('Error: Invalid results')

    if torch.isnan(anno_bb).any():
        if dataset == 'uav':
            pass
        else:
            raise Exception('Warning: NaNs in annotation')

    if (pred_bb[:, 2:] == 0.0).any():
        for i in range(1, pred_bb.shape[0]):
            if (pred_bb[i, 2:] == 0.0).any() and not torch.isnan(anno_bb[i, :]).any():
                pred_bb[i, :] = pred_bb[i-1, :]

    if pred_bb.shape[0] != anno_bb.shape[0]:
        if dataset == 'lasot':
            if pred_bb.shape[0] > anno_bb.shape[0]:
                # For monkey-17, there is a mismatch for some trackers.
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                raise Exception('Mis-match in tracker prediction and GT lengths')
        else:
            print('Warning: Mis-match in tracker prediction and GT lengths')
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                pad = torch.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4)).type_as(pred_bb)
                pred_bb = torch.cat((pred_bb, pad), dim=0)

    if target_visible is not None:
        target_visible = target_visible.bool()
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    # handle invalid anno cases
    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot':
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid


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


def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)
    return auc_curve, auc


def get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]
    return prec_curve, prec_score


def tlbr_to_tlwh(tlbr):
    # 计算边界框的宽度和高度
    width = tlbr[:, 2] - tlbr[:, 0]
    height = tlbr[:, 3] - tlbr[:, 1]

    # 转换为TLWH表示
    tlwh = torch.stack([tlbr[:, 0], tlbr[:, 1], width, height], dim=1).clamp(1, 100)
    return tlwh


def extract_results(filename, plot_bin_gap=0.05, vis=False, exclude_invalid_frames=False):

    with open(filename) as f:
        flat_outputs = [json.loads(line) for line in f]
    print(len(flat_outputs))

    source_case_map = {}

    for i, item in enumerate(flat_outputs):
        item["source"] = item.get("source", "unknown")
        if item["source"] not in source_case_map:
            source_case_map[item["source"]] = []
        source_case_map[item["source"]].append(item)

    for key, value in source_case_map.items():
        print(key)
        trackers = [1]
        dataset = [1] * len(value)
        threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
        threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
        threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

        avg_overlap_all = torch.zeros((len(dataset), len(trackers)), dtype=torch.float64)
        ave_success_rate_plot_overlap = torch.zeros((len(dataset), len(trackers), threshold_set_overlap.numel()),
                                                    dtype=torch.float32)
        ave_success_rate_plot_center = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                                dtype=torch.float32)
        ave_success_rate_plot_center_norm = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                                        dtype=torch.float32)
        trk_id = 0    
        valid_sequence = []                                        
        for seq_id, item in enumerate(value):
            w, h = item["image_size"]
            scale_tenosr = torch.tensor([w, h, w, h]) / 100
            pred_bb = torch.tensor(parse_box_from_raw_text(item["predict"])) * scale_tenosr
            anno_bb = torch.tensor(parse_box_from_raw_text(item["gt"])) * scale_tenosr
            # pred_bb = torch.tensor(parse_box_from_raw_text(item["predict"]))
            # anno_bb = torch.tensor(parse_box_from_raw_text(item["gt"]))
            if len(pred_bb) < 1:
                continue
            if len(pred_bb[0]) < 4:
                continue
            err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(
                tlbr_to_tlwh(pred_bb), tlbr_to_tlwh(anno_bb), "ours", target_visible=None)
            print(err_overlap, err_center, err_center_normalized, valid_frame)
            avg_overlap_all[seq_id, trk_id] = err_overlap[valid_frame].mean()
            if exclude_invalid_frames:
                seq_length = valid_frame.long().sum()
            else:
                seq_length = anno_bb.shape[0]

            if seq_length <= 0:
                raise Exception('Seq length zero')

            ave_success_rate_plot_overlap[seq_id, trk_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center[seq_id, trk_id, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center_norm[seq_id, trk_id, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length
            valid_sequence.append(seq_id)
            # ********************************  Success Plot **************************************
        valid_sequence = torch.tensor(valid_sequence)
        # ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        # threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])

        # success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
        #                      'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 88), 'title': 'Success'}
        # plot_draw_save(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path, success_plot_opts)

    # ********************************  Precision Plot **************************************

        # ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        # threshold_set_center = torch.tensor(eval_data['threshold_set_center'])

        # precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
        #                        'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision [%]',
        #                        'xlim': (0, 50), 'ylim': (0, 100), 'title': 'Precision plot'}
        # plot_draw_save(prec_curve, threshold_set_center, prec_score, tracker_names, plot_draw_styles, result_plot_path,
        #                precision_plot_opts)

    # ********************************  Norm Precision Plot **************************************
    # if 'norm_prec' in plot_types:
        # ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        # threshold_set_center_norm = torch.tensor(eval_data['threshold_set_center_norm'])

        # norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
        #                             'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
        #                             'xlim': (0, 0.5), 'ylim': (0, 85), 'title': 'Normalized Precision'}
        print("auc: ", auc, )
        print("prec_score: ", prec_score, )
        print("norm_prec_score: ", norm_prec_score, )


if __name__ == '__main__':

    argument_parser = ArgumentParser()
    argument_parser.add_argument('file', type=str, help='Path to the file containing the results')
    args = argument_parser.parse_args()

    extract_results(args.file)