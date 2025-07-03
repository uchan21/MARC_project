import rclpy
import json
from rclpy.node import Node
from rclpy.qos import QoSProfile

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses

detections = []

bridge = CvBridge()
img = None

class LastTopicNode(Node):

    def __init__(self) -> None:
        super().__init__("last_topic_data_node")
        qos = QoSProfile(depth = 10)
        self.extraction_BB = self.create_subscription(DetectionArray, '/yolo/gwanggaeto_2/detections',self.callback_Bb, qos)
        self.extraction_im = self.create_subscription(Image, '/metasejong2025/cameras/gwanggaeto_2/image_raw', self.callback_Im, qos)
        self.get_logger().info('Starting last Topic data node')

    def callback_Bb(self, msg):
        global detections
        detections = []
        for det in msg.detections:
            class_id = det.class_id
            class_name = det.class_name
            x_min = det.bbox.center.position.x - det.bbox.size.x/2
            x_max = det.bbox.center.position.x + det.bbox.size.x/2
            y_min = det.bbox.center.position.y - det.bbox.size.y/2
            y_max = det.bbox.center.position.y + det.bbox.size.y/2

            detections.append([class_id, class_name, [x_min, y_min], [x_max, y_max]])

    def callback_Im(self, msg):
        global img
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(str(e))


def main(args=None):
    global img, detections
    rclpy.init(args=args) #초기화
    node = LastTopicNode() # 노드 선언
    try:
        rclpy.spin(node) # 노드 spin
    except KeyboardInterrupt: # ctrl + C 키 입력 시
        with open('output.json','w',encoding='utf-8') as f:
            json.dump(detections, f, ensure_ascii=False ,indent=2)
        filename = f'output_image.jpg'
        cv2.imwrite(filename, img)
    finally:
        node.destroy_node() # 노드 제거
        #rclpy.shutdown() # 종료