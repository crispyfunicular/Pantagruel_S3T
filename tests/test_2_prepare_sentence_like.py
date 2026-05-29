from __future__ import annotations

from pathlib import Path

from conftest import load_stage_module

prepare = load_stage_module("2_prepare.py")
SegmentRecord = prepare.SegmentRecord
merge_segments_sentence_like = prepare.merge_segments_sentence_like


def _seg(
    *,
    utt_id: str,
    talk_id: str,
    order_idx: int,
    offset_s: float,
    duration_s: float,
    src_text: str,
    tgt_text: str,
    speaker: str = "spk",
) -> SegmentRecord:
    return SegmentRecord(
        utt_id=utt_id,
        talk_id=talk_id,
        order_idx=order_idx,
        wav_path=Path("/tmp/fake.flac"),
        offset_s=offset_s,
        duration_s=duration_s,
        src_text=src_text,
        tgt_text=tgt_text,
        speaker=speaker,
        src_lang="fr",
        tgt_lang="en",
    )


def test_sentence_like_merges_until_punctuation_and_target_duration() -> None:
    segs = [
        _seg(
            utt_id="t_0",
            talk_id="t",
            order_idx=0,
            offset_s=0.0,
            duration_s=4.0,
            src_text="Bonjour",
            tgt_text="Hello",
        ),
        _seg(
            utt_id="t_1",
            talk_id="t",
            order_idx=1,
            offset_s=4.0,
            duration_s=3.0,
            src_text="comment ça va ?",
            tgt_text="how are you?",
        ),
        _seg(
            utt_id="t_2",
            talk_id="t",
            order_idx=2,
            offset_s=7.0,
            duration_s=2.0,
            src_text="Merci.",
            tgt_text="Thanks.",
        ),
    ]
    merged, stats = merge_segments_sentence_like(
        segs, target_duration_s=6.0, max_duration_s=15.0, require_punctuation=True
    )
    assert stats["segments_in"] == 3
    # Le premier groupe doit inclure seg0+seg1 (punctuation ? + durée >= 6s).
    assert len(merged) == 2
    assert merged[0].src_text.endswith("?")
    assert abs(merged[0].duration_s - 7.0) < 1e-6
    assert merged[1].src_text.endswith(".")


def test_sentence_like_does_not_merge_across_talks() -> None:
    segs = [
        _seg(
            utt_id="a_0",
            talk_id="a",
            order_idx=0,
            offset_s=0.0,
            duration_s=5.0,
            src_text="A.",
            tgt_text="A.",
        ),
        _seg(
            utt_id="b_0",
            talk_id="b",
            order_idx=0,
            offset_s=0.0,
            duration_s=5.0,
            src_text="B.",
            tgt_text="B.",
        ),
    ]
    merged, _ = merge_segments_sentence_like(
        segs, target_duration_s=6.0, max_duration_s=15.0, require_punctuation=True
    )
    assert len(merged) == 2
    assert merged[0].talk_id == "a"
    assert merged[1].talk_id == "b"


def test_sentence_like_respects_max_duration() -> None:
    segs = [
        _seg(
            utt_id="t_0",
            talk_id="t",
            order_idx=0,
            offset_s=0.0,
            duration_s=10.0,
            src_text="Long segment without punctuation",
            tgt_text="Long segment without punctuation",
        ),
        _seg(
            utt_id="t_1",
            talk_id="t",
            order_idx=1,
            offset_s=10.0,
            duration_s=10.0,
            src_text="still going",
            tgt_text="still going",
        ),
    ]
    merged, _ = merge_segments_sentence_like(
        segs, target_duration_s=10.0, max_duration_s=15.0, require_punctuation=True
    )
    # 10 + 10 > 15 => pas de fusion.
    assert len(merged) == 2
