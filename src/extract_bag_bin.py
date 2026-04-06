import rosbag2_py
import cv2
import sys
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message


def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def process_bag(bag_path, save_path, start_count):
    """
    处理单个 bag 并返回结束时的图片计数，确保多个 bag 图片命名连续。
    """
    storage_options, converter_options = get_rosbag_options(bag_path)

    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"无法打开 {bag_path}: {e}")
        return start_count

    storage_filter = rosbag2_py.StorageFilter(
        topics=['/detector/img_armor_processed'])
    reader.set_filter(storage_filter)

    bridge = CvBridge()
    count = start_count

    print(f"正在处理: {bag_path}")
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg = deserialize_message(data, Image)

        # 转换为 OpenCV 图像
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        out_file = os.path.join(save_path, "%06i.png" % count)
        cv2.imwrite(out_file, cv_img)

        # 如果输出日志过多影响性能，可以将下面这行注释掉
        print("Writing binarized image %i" % count)
        count += 1

    print(f"完成 {bag_path}! 本次提取了 {count - start_count} 张图片。")
    return count


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_images.py <rosbag_path> <save_path>")
        return

    base_bag_path = sys.argv[1]
    save_path = sys.argv[2]

    # 自动创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 1. 自动寻找子数据集路径
    bag_paths_to_process = []
    i = 1
    while True:
        sub_bag_path = f"{base_bag_path}.{i}"
        if os.path.exists(sub_bag_path):
            bag_paths_to_process.append(sub_bag_path)
            i += 1
        else:
            break
    
    # 2. 如果没有找到类似 .1, .2 的后缀路径，则处理传入的基础路径
    if not bag_paths_to_process:
        if os.path.exists(base_bag_path):
            bag_paths_to_process.append(base_bag_path)
        else:
            print(f"错误: 找不到路径 {base_bag_path}")
            return

    print(f"共发现 {len(bag_paths_to_process)} 个 bag 需要处理。")
    print("Saving binarized images to %s" % save_path)

    # 3. 遍历处理并保持计数连续
    total_count = 0
    for bag_path in bag_paths_to_process:
        total_count = process_bag(bag_path, save_path, total_count)

    print("Finished! Total saved:", total_count)


if __name__ == '__main__':
    main()