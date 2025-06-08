import easyocr

def get_ocr_docs(frames):
    reader = easyocr.Reader(['en']) 
    text_set = []
    ocr_docs = []
    for img in frames:
        ocr_results = reader.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 and text not in text_set:
                det_info += f"{text}; "
                text_set.append(text)
        if len(det_info) > 0:
            ocr_docs.append(det_info)

    return ocr_docs