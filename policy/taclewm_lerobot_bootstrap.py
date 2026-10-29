"""Launch the native Tac-LeWM trainer with a LeRobot dataset adapter."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "policy/TacMind-0"
SOURCE = MODEL_SOURCE / "tacmind0/tactile/lewm/pretraining/upstream"
VENDOR = SOURCE / "third_party/le-wm"
sys.path[:0] = [str(ROOT), str(MODEL_SOURCE), str(SOURCE), str(VENDOR), str(VENDOR / "stable-worldmodel")]


def main():
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from tacbenchhub.models.lewm import visuo_jepa_train as native
    from policy.adapters import temporary_attribute
    from policy.taclewm_lerobot import LeRobotWorldDataset

    original_split = spt.data.random_split

    def load_dataset(name, *, transform=None, cache_dir=None, **kwargs):
        root = Path(str(name))
        dataset = LeRobotWorldDataset(root, history_size=1, **kwargs)
        dataset.transform = transform
        return dataset

    def split(dataset, lengths, generator):
        if not isinstance(dataset, LeRobotWorldDataset):
            return original_split(dataset, lengths=lengths, generator=generator)
        train, val = native.stratified_episode_split(
            dataset.task_ids, train_split=float(lengths[0]), seed=int(generator.initial_seed()))
        return (native.subset_clips_by_episode(dataset, set(train)),
                native.subset_clips_by_episode(dataset, set(val)))

    # The converted corpus has one action source, as did its HDF5 predecessor.
    with (
        temporary_attribute(swm.data, "load_dataset", load_dataset),
        temporary_attribute(spt.data, "random_split", split),
        temporary_attribute(native, "_active_source_ids_from_h5", lambda cfg, h5_path: {0}),
    ):
        native.run()


if __name__ == "__main__":
    main()
