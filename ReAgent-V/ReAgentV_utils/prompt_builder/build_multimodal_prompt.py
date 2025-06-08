from typing import Dict, List


def build_multimodal_prompt(
    text: str,
    modal_info: Dict[str, List[str]],
    det_top_idx: List[int],
    max_frames_num: int,
    USE_DET: bool = False,
    USE_ASR: bool = False,
    USE_OCR: bool = False
) -> str:
   
    ocr_docs = modal_info.get("OCR", [])
    asr_docs = modal_info.get("ASR", [])
    det_docs = modal_info.get("DET", [])
   
    qs = ""

    if USE_DET and len(det_docs) > 0:
        for i, info in enumerate(det_docs):
            if len(info) > 0:
                qs += f"Frame {str(det_top_idx[i] + 1)}: " + info + "\n"
        if len(qs) > 0:
            qs = (
                f"\nVideo have {str(max_frames_num)} frames in total, "
                "the detected objects' information in specific frames: "
                + qs
            )

    if USE_ASR and len(asr_docs) > 0:
        qs += (
            "\nVideo Automatic Speech Recognition information "
            "(given in chronological order of the video): "
            + " ".join(asr_docs)
        )

    if USE_OCR and len(ocr_docs) > 0:
        qs += (
            "\nVideo OCR information "
            "(given in chronological order of the video): "
            + "; ".join(ocr_docs)
        )

    qs += (
        " Based on the video and the information (if given), provide a concise, direct answer to the following question in free-form natural language. "
        "Do not include option labels (A, B, etc.). Respond with a complete sentence or phrase.\n"
        "Question: " + text + "\nAnswer:"
    )

    return qs


