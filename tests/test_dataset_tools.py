import xml.etree.ElementTree as ET

import cv2
import numpy as np

from prepare_dataset import prepare
from voc_to_yolo import convert_file, voc_box_to_yolo


def write_xml(path, objects):
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "100"
    ET.SubElement(size, "height").text = "100"
    for name, box in objects:
        item = ET.SubElement(root, "object")
        ET.SubElement(item, "name").text = name
        bounds = ET.SubElement(item, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bounds, key).text = str(value)
    ET.ElementTree(root).write(path)


def test_voc_geometry():
    assert voc_box_to_yolo((10, 20, 30, 60), 100, 100) == (0.2, 0.4, 0.2, 0.4)


def test_voc_file_conversion_uses_dynamic_classes(tmp_path):
    xml = tmp_path / "sample.xml"
    out = tmp_path / "sample.txt"
    write_xml(xml, [("visor", (10, 20, 30, 60))])
    assert convert_file(xml, out, ["helmet", "visor"]) == 1
    assert out.read_text().startswith("1 0.200000 0.400000")


def test_prepare_dataset_creates_owned_child_crop(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "images").mkdir(parents=True)
    (source / "annotations").mkdir()
    cv2.imwrite(str(source / "images" / "sample.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
    write_xml(source / "annotations" / "sample.xml", [("worker", (10, 10, 90, 90)), ("visor", (30, 30, 50, 50))])
    counts = prepare(source, output, "worker", ["visor"], padding=0, val_ratio=0)
    label = output / "child" / "labels" / "train" / "sample_p000.txt"
    assert counts == {"source_images": 1, "parent_crops": 1, "negative_crops": 0}
    assert label.read_text().strip() == "0 0.375000 0.375000 0.250000 0.250000"
    assert (output / "parent" / "images" / "train" / "sample.jpg").is_file()

