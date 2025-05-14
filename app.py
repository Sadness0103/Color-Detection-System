from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np

app = Flask(__name__)

# Khởi tạo camera
camera = cv2.VideoCapture(0)

# Định nghĩa phạm vi màu sắc (HSV)
color_ranges = {
    'red': ([0, 100, 100], [10, 255, 255]),
    'green': ([40, 100, 100], [80, 255, 255]),
    'blue': ([100, 100, 100], [140, 255, 255])
}

# Biến đếm màu
color_counts = {'red': 0, 'green': 0, 'blue': 0}
# Biến lưu trạng thái hiện tại của các màu
current_positions = {'red': [], 'green': [], 'blue': []}
# Biến lưu trạng thái đã đếm của các màu
counted_objects = {'red': False, 'green': False, 'blue': False}

# Định nghĩa vùng phát hiện màu ban đầu (x, y, width, height)
DETECTION_ZONE = {
    'x': 220,  # Vị trí x ban đầu
    'y': 140,  # Vị trí y ban đầu
    'width': 200,  # Chiều rộng ban đầu
    'height': 200   # Chiều cao ban đầu
}

def is_in_detection_zone(x, y, w, h):
    # Kiểm tra xem vùng màu có nằm trong vùng phát hiện không
    if DETECTION_ZONE is None:
        return False
        
    center_x = x + w/2
    center_y = y + h/2
    
    return (DETECTION_ZONE['x'] <= center_x <= DETECTION_ZONE['x'] + DETECTION_ZONE['width'] and
            DETECTION_ZONE['y'] <= center_y <= DETECTION_ZONE['y'] + DETECTION_ZONE['height'])

def detect_colors(frame):
    # Chuyển đổi BGR sang HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Reset current positions mỗi frame
    for color in current_positions:
        current_positions[color] = []
    
    # Vẽ vùng phát hiện màu
    cv2.rectangle(frame, 
                 (DETECTION_ZONE['x'], DETECTION_ZONE['y']),
                 (DETECTION_ZONE['x'] + DETECTION_ZONE['width'], 
                  DETECTION_ZONE['y'] + DETECTION_ZONE['height']),
                 (52, 152, 219), 2)  # Màu xanh dương
    
    # Kiểm tra từng màu
    for color, (lower, upper) in color_ranges.items():
        # Tạo mask cho màu
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Vẽ contour cho màu được phát hiện
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Lọc nhiễu
                x, y, w, h = cv2.boundingRect(contour)
                
                # Chỉ xử lý nếu vùng màu nằm trong vùng phát hiện
                if is_in_detection_zone(x, y, w, h):
                    # Lưu vị trí hiện tại
                    current_positions[color].append((x, y, w, h))
                    
                    # Nếu có vật thể trong vùng phát hiện và chưa đếm
                    if len(current_positions[color]) > 0 and not counted_objects[color]:
                        color_counts[color] += 1
                        counted_objects[color] = True
                    
                    # Vẽ khung và tên màu
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{color}: {color_counts[color]}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    # Reset trạng thái đếm khi vật thể ra khỏi vùng phát hiện
                    counted_objects[color] = False
    
    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_colors(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('web.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/color_counts')
def get_color_counts():
    return jsonify(color_counts)

@app.route('/update_zone', methods=['POST'])
def update_zone():
    try:
        data = request.get_json()
        global DETECTION_ZONE
        
        # Cập nhật vùng phát hiện
        DETECTION_ZONE = {
            'x': max(0, min(data['x'], 640)),
            'y': max(0, min(data['y'], 480)),
            'width': max(50, min(data['width'], 640)),
            'height': max(50, min(data['height'], 480))
        }
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 