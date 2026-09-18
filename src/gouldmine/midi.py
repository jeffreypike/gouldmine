"""
Convert agent outputs to songs we can hear.
"""

import numpy as np
import pretty_midi


def array_to_midi(
    song: np.ndarray,
    num_actions: int = 14,
    bpm: int = 120,
    step_duration: float = 0.125,
) -> pretty_midi.PrettyMIDI:
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )

    current_pitch = None
    start_time = 0.0
    end_step = len(song)

    for t, action in enumerate(song):
        if action < num_actions - 2:
            if current_pitch is not None:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=current_pitch,
                    start=start_time,
                    end=t * step_duration,
                )
                piano.notes.append(note)

            current_pitch = 60 + action
            start_time = t * step_duration

        elif action == num_actions - 2 and current_pitch is not None:
            note = pretty_midi.Note(
                velocity=100,
                pitch=current_pitch,
                start=start_time,
                end=t * step_duration,
            )
            piano.notes.append(note)
            current_pitch = None

        elif action == num_actions:
            end_step = t
            break

    if current_pitch is not None:
        note = pretty_midi.Note(
            velocity=100,
            pitch=current_pitch,
            start=start_time,
            end=end_step * step_duration,
        )
        piano.notes.append(note)

    midi_data.instruments.append(piano)
    return midi_data


def save_midi(midi_data: pretty_midi.PrettyMIDI, output_path: str):
    midi_data.write(output_path)
