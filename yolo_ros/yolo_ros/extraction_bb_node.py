import rclpy
import json
from rclpy.qos import QoSProfile

from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses

from rclpy.node import Node

images = []

class ExtractionBbNode(Node):

    def __init__(self) -> None:
        super().__init__("extraction_BB")
        qos = QoSProfile(depth = 10)
        self.extraction_BB = self.create_subscription(DetectionArray, '/yolo/demo_2/detections',self.callback, qos)
        self.get_logger().info('Starting Extraction_BB_node')

    def callback(self, msg):
        detections = []
        for det in msg.detections:
            class_id = det.class_id
            class_name = det.class_name
            x_min = det.bbox.center.position.x - det.bbox.size.x/2
            x_max = det.bbox.center.position.x + det.bbox.size.x/2
            y_min = det.bbox.center.position.y - det.bbox.size.y/2
            y_max = det.bbox.center.position.y + det.bbox.size.y/2

            detections.append([class_id, class_name, [x_min, y_min], [x_max, y_max]])
        images.append(detections)

def main(args=None):
    rclpy.init(args=args) #초기화
    node = ExtractionBbNode() # 노드 선언
    try:
        rclpy.spin(node) # 노드 spin
    except KeyboardInterrupt: # ctrl + C 키 입력 시
        with open('output.json','w',encoding='utf-8') as f:
            json.dump(images, f, ensure_ascii=False ,indent=2)
    finally:
        node.destroy_node() # 노드 제거
        #rclpy.shutdown() # 종료


if __name__ == '__main__':
    main()