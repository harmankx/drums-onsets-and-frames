import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval import transcription
from mir_eval import transcription_velocity
from mir_eval import util
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm

import onsets_and_frames.dataset as dataset_module
from onsets_and_frames import *

eps = sys.float_info.epsilon

def split_midi(pitches, intervals, velocity, midi):
    """
    Takes extracted pitch values, intervals, and velocitys and filters by a midi hit
    """
    combined = np.concatenate((pitches[:, None], intervals.reshape(-1, 2), velocity[:, None]), axis=1)
    midi_combined = combined[combined[:, 0] == midi]
    p = midi_combined[:, 0].reshape(-1)
    i = midi_combined[:, [1, 2]]
    v = midi_combined[:, 3].reshape(-1)
    return p, i, v

def evaluate(data, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    metrics = defaultdict(list)

    for label in data:
        pred, losses = model.run_on_batch(label)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            value.squeeze_(0).relu_()

        p_ref, i_ref, v_ref = extract_notes(label['onset'], label['velocity'])
        p_est, i_est, v_est = extract_notes(pred['onset'], pred['velocity'], onset_threshold)

        scaling = HOP_LENGTH / SAMPLE_RATE
        for i in range(8):
            hit = i + 1
            pitch_ref, int_ref, vel_ref = split_midi(p_ref, i_ref, v_ref, hit)
            pitch_est, int_est, vel_est = split_midi(p_est, i_est, v_est, hit)
            if(pitch_ref.size == 0):
                continue
            int_ref = (int_ref * scaling).reshape(-1, 2)
            pitch_ref = np.array([midi_to_hz(HIT_MAPS_ENCODE[midi]) for midi in pitch_ref])
            int_est = (int_est * scaling).reshape(-1, 2)
            pitch_est = np.array([midi_to_hz(HIT_MAPS_ENCODE[midi]) for midi in pitch_est])

            p, r, f, o = evaluate_notes(int_ref, pitch_ref, int_est, pitch_est, offset_ratio=None)
            metrics['metric/' + str(HIT_MAPS_NAMES[hit]) + '/f1'].append(f)

            p, r, f, o = evaluate_notes_with_velocity2(int_ref, pitch_ref, vel_ref, int_est, pitch_est, vel_est,
                                                    offset_ratio=None, velocity_tolerance=0.1)
            metrics['metric/' + str(HIT_MAPS_NAMES[hit]) + '-with-velocity/f1'].append(f)

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(HIT_MAPS_ENCODE[midi]) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(HIT_MAPS_ENCODE[midi]) for midi in p_est])

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/total/precision'].append(p)
        metrics['metric/total/recall'].append(r)
        metrics['metric/total/f1'].append(f)
        onset_f = f

        p, r, f, o = evaluate_notes_with_velocity2(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                  offset_ratio=None, velocity_tolerance=0.1)
        metrics['metric/total-with-velocity/precision'].append(p)
        metrics['metric/total-with-velocity/recall'].append(r)
        metrics['metric/total-with-velocity/f1'].append(f)
        metrics['path'].append(label["path"])

        # if not model.training:
        #     print(f'f1: {str(onset_f):30s} velocity/f1: {str(f):30s} file: {str(label["path"]):30s}')

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
            save_pianoroll(label_path, label['onset'])
            pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['onset'])
            midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
            save_midi(midi_path, p_est, i_est, v_est)

    return metrics


def evaluate_file(model_file, dataset, dataset_group, sequence_length, save_path,
                  onset_threshold, frame_threshold, device):
    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length, 'device': device}
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    metrics = evaluate(tqdm(dataset), model, onset_threshold, frame_threshold, save_path)

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')


def evaluate_notes_with_velocity2(ref_intervals, ref_pitches, ref_velocities, est_intervals, est_pitches,
        est_velocities, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05, strict=False,
        velocity_tolerance=0.1, beta=1.0):
    transcription_velocity.validate(ref_intervals, ref_pitches, ref_velocities, est_intervals,
             est_pitches, est_velocities)
    # When reference notes are empty, metrics are undefined, return 0's
    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return 0., 0., 0., 0.

    matching = match_notes(
        ref_intervals, ref_pitches, ref_velocities, est_intervals, est_pitches,
        est_velocities, onset_tolerance, pitch_tolerance, offset_ratio,
        offset_min_tolerance, strict, velocity_tolerance)

    precision = float(len(matching))/len(est_pitches)
    recall = float(len(matching))/len(ref_pitches)
    f_measure = util.f_measure(precision, recall, beta=beta)

    avg_overlap_ratio = transcription.average_overlap_ratio(
        ref_intervals, est_intervals, matching)
    return precision, recall, f_measure, avg_overlap_ratio

def match_notes(
        ref_intervals, ref_pitches, ref_velocities, est_intervals, est_pitches,
        est_velocities, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05, strict=False,
        velocity_tolerance=0.1):

    # Compute note matching as usual using standard transcription function
    matching = transcription.match_notes(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance, pitch_tolerance, offset_ratio, offset_min_tolerance,
        strict)

    # Convert matching list-of-tuples to array for fancy indexing
    matching = np.array(matching)
    # When there is no matching, return an empty list
    if matching.size == 0:
        return []
    # Grab velocities for matched notes
    ref_matched_velocities = ref_velocities[matching[:, 0]]
    est_matched_velocities = est_velocities[matching[:, 1]]

    velocity_diff = np.abs(est_matched_velocities - ref_matched_velocities)
    # Check whether each error is within the provided tolerance
    velocity_within_tolerance = (velocity_diff < velocity_tolerance)
    # Only keep matches whose velocity was within the provided tolerance
    matching = matching[velocity_within_tolerance]
    # Convert back to list-of-tuple format
    matching = [tuple(_) for _ in matching]

    return matching

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', nargs='?', type=str, default='runs/transcriber-CE-20/model-50000.pt')
    parser.add_argument('dataset', nargs='?', default='GROOVE')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sequence-length', default=SEQUENCE_LENGTH, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        evaluate_file(**vars(parser.parse_args()))