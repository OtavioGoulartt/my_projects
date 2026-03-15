import numpy as np
from fs_msgs.msg import TrackStampedWithCovariance, Track, ConeWithCovariance
from sensor_msgs.msg import Image
import os
import yaml
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()

class PerceptionProcess:
    # perception_calc(endereço_arq_yaml, disp_img).triangulacao(baseline,yoloinference) = ((X,Y,Z)) -> Posicao do cone no espaco 3D.
    def __init__(self, baseline):

        endereco_matriz_intrinsica_left = '/home/otaviogoulart/ws/src/amp_perception/perception/config/OAKDLR_left.yaml'
        endereco_matriz_intrinsica_right = '/home/otaviogoulart/ws/src/amp_perception/perception/config/OAKDLR_right.yaml'
        self.camera_matrix = self.yaml_reader(endereco_matriz_intrinsica_left, endereco_matriz_intrinsica_right)

            #OAK D LR
        self.focal_length_x = self.camera_matrix[0][0][0][0]
        self.focal_length_y = self.camera_matrix[0][0][1][1]
        self.center_x = self.camera_matrix[0][0][0][2]
        self.center_y = self.camera_matrix[0][0][1][2]

            #OAK D W
        #self.focal_length_x = 574.92
        #self.focal_length_y = 574.92
        #self.center_x = 639.78
        #self.center_y = 377.21

        self.baseline = baseline

    def object_on_map(self, yoloinference, disp_map, imgL_raw_ros_msg, imgR_raw_ros_msg):  
        
        height = imgL_raw_ros_msg.height
        width = imgL_raw_ros_msg.width

        imgL_raw_ros_msg = bridge.imgmsg_to_cv2(imgL_raw_ros_msg)
        imgR_raw_ros_msg = bridge.imgmsg_to_cv2(imgR_raw_ros_msg)
        
        cone_list = []
        bb_yolo = yoloinference.yolov8_inference
        
        for box in bb_yolo:
            cone = ConeWithCovariance()
            cor = box.class_name
            confidence = box.confidence
            
            if cor == 'blue_cone':
                cone.color=0
            elif cor == 'yellow_cone':
                cone.color=1
            elif cor == 'large_orange_cone':
                cone.color=2
            
            x1 = box.top
            y1 = box.left
            x2 = box.bottom
            y2 = box.right
            
            center_y = int((abs(y2-y1) / 2) + min(y1, y2))
            center_x = int((abs(x2-x1) / 2) + min(x1, x2))
            
            bb_w = x2-x1
            bb_h = y2-y1
            
            sample_w = max(1, (bb_w * 0.15)//2)
            sample_h = max(1, (bb_h * 0.2)//2)
            
            obj_x1 = int(max(0, center_x - sample_w))
            obj_y1 = int(max(0, center_y - sample_h))
            obj_x2 = int(min(width, center_x + sample_w))
            obj_y2 = int(min(height, center_y + sample_h))
            
            roi = disp_map[obj_y1:obj_y2, obj_x1:obj_x2]
            valid = roi[np.isfinite(roi)]
            valid = valid[valid > 0]
            
            median_disp = np.median(valid)
            
            if len(valid) >= 1:
                if median_disp > 300:
                    Z = median_disp / 1000
                    X, Y = self.x_y_space_measure(Z, center_x, center_y)
                    is_disp_map = True

                else:
                    X,Y,Z = self.triangulacao(center_y, center_x, median_disp, disp_map, imgL_raw_ros_msg, imgR_raw_ros_msg)
                    is_disp_map = False

            cone.location.x = X
            cone.location.y = Y
            cone.location.z = Z
            
            deviationZ = 0.0096*cone.location.z + 0.1643   #linearização do erro da detecção vs distancia no eixo z
            deviationX = 0.0232*cone.location.x + 0.1204   #linearização do erro da detecção vs distancia no eixo x
            deviation = np.sqrt(deviationX**2 + deviationZ**2)   
            
            cone.deviation = deviation
            cone.confidence = confidence
            cone_list.append(cone)
              
        cone_track = TrackStampedWithCovariance()
        cone_track.track = cone_list

        return (cone_track, is_disp_map)


    def x_y_space_measure(self, Z_point, center_x, center_y):
        X_point = (center_x-self.center_x) * Z_point / self.focal_length_x
        Y_point = (center_y-self.center_y) * Z_point / self.focal_length_y  
        return X_point, Y_point
        
    def triangulacao(self, center_y, center_x, disparity, disp_map, imgL_raw_ros_msg, imgR_raw_ros_msg):
        
        Z = (self.baseline * self.focal_length_x ) / disparity
        Z = Z 
        X, Y = self.x_y_space_measure(Z, center_x, center_y)
        
        if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z):
            return float(X), float(Y), float(Z)
        else:
            return 0.0, 0.0, 0.0

    def triangulacao_lux(self, center_y, center_x, disparity, disp_map, imgL_raw_ros_msg, imgR_raw_ros_msg):
        
        Z = (self.baseline * self.focal_length_x ) / disparity
        Z = Z *16
        X, Y = self.x_y_space_measure(Z, center_x, center_y)
        
        if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z):
            return float(X), float(Y), float(Z)
        else:
            return 0.0, 0.0, 0.0

    def DisparityProcess(self, imgL_ros_msg, imgR_ros_msg):
        
        imgL_cv = bridge.imgmsg_to_cv2(imgL_ros_msg)
        imgR_cv = bridge.imgmsg_to_cv2(imgR_ros_msg)

        #deixa a imagem em preto e branco.
        #imgL_cv = cv2.cvtColor(imgL_cv,cv2.COLOR_BGR2GRAY)
        #imgR_cv = cv2.cvtColor(imgR_cv,cv2.COLOR_BGR2GRAY)

        # Creating an object of StereoSGBM algorithm
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*11,
            blockSize=7,
            P1=8*3*7**2,
            P2=32*3*7**2,   
            disp12MaxDiff=12,
            uniquenessRatio=3,
            speckleWindowSize=100,
            speckleRange=64,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        # Calculating disparith using the StereoSGBM algorithm
        stereo = stereo.compute(imgL_cv, imgR_cv).astype(np.float32) / 16

        # Calculating disparith using the StereoSGBM algorithm

        disp_map = cv2.normalize(stereo,None, 0, 255, cv2.NORM_MINMAX)
        disp_map = np.uint8(disp_map)
        #disp_map[disp_map < 0] = 0
        #disp_map[disp_map > 64] = 64
        #disp_vis = (disp_map / np.max(disp_map) * 255).astype(np.uint8)
        #disp_vis = cv2.medianBlur(disp_vis, 5)  

        return (disp_map, stereo)
    
    def yaml_reader(self, endereco_left, endereco_right):
        try:
            saida = [None, None]
            with open(endereco_left, 'r') as f:
                data = yaml.safe_load(f)

                kL = np.array(data['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
                dL = np.array(data['distortion_coefficients']['data'], dtype=np.float64)
                rL = np.array(data['rectification_matrix']['data'], dtype=np.float64).reshape(3, 3)
                pL_raw = np.array(data['projection_matrix']['data'], dtype=np.float64).reshape(3, 4)

                pL = np.zeros((3, 4), dtype=np.float64)
                pL[0:3, 0:3] = kL
                pL[0, 3] = kL[0, 0] * pL_raw[0, 3]
                pL[1, 3] = kL[1, 1] * pL_raw[1, 3]
                pL[2, 3] = pL_raw[2, 3]

                saida[0] = (kL, dL, rL, pL)
                
            with open(endereco_right, 'r') as f:
                data = yaml.safe_load(f)

                kR = np.array(data['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
                dR = np.array(data['distortion_coefficients']['data'], dtype=np.float64)
                rR = np.array(data['rectification_matrix']['data'], dtype=np.float64).reshape(3, 3)
                pR_raw = np.array(data['projection_matrix']['data'], dtype=np.float64).reshape(3, 4)

                pR = np.zeros((3, 4), dtype=np.float64)
                pR[0:3, 0:3] = kR
                pR[0, 3] = kR[0, 0] * pR_raw[0, 3]
                pR[1, 3] = kR[1, 1] * pR_raw[1, 3]
                pR[2, 3] = pR_raw[2, 3]

                saida[1] = (kR, dR, rR, pR)

            return saida
                
        except FileNotFoundError:
            print("ERRO: Arquivo YAML não encontrado no endereco")
            return None
        except KeyError:
            print("ERRO: Palavra-chave não encontrada no arquivo")
            return None
        except Exception as e:
            print("ERRO inesperado ao ler YAML: {e}")
            return None