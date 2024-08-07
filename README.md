---

# Sentinel Vision Dynamics 🚀

Sentinel Vision Dynamics is an advanced real-time object detection and notification system. Leveraging the power of YOLO v8 by Ultralytics, Supervision by Roboflow, and various Python libraries, this project integrates with live video feed sources like CCTV to detect harmful or dangerous objects. Upon detection, it immediately notifies the user or the provided authority through SMS and system notifications.

## 📋 Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## 🌟 Features

- **Real-Time Object Detection**: Utilizes YOLO v8 for efficient and accurate object detection in live video feeds.
- **Integration with CCTV**: Connects seamlessly with live video feed sources for continuous monitoring.
- **Immediate Notifications**: Sends real-time alerts via SMS using Twilio and system notifications using win10toast.
- **User-Friendly**: Easy to set up and configure, making it accessible for various use cases.

## 💼 Technologies Used

- **YOLO v8** by Ultralytics
- **Supervision** by Roboflow - Reusable Computer Vision Tools
- **Python Libraries**: OpenCV (cv2), argparse, NumPy, win10toast, Twilio

## 💻 Installation

Follow these steps to set up the project on your local machine:

### Install

```bash
# create python virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### 📸 Execute

```bash
python3 -m main
```

## 🚀 Getting Started

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Milind-Palaria/Sentinel-Vision-Dynamics.git
    cd Sentinel-Vision-Dynamics
    ```

2. **Prepare your environment**: Follow the installation steps provided above to set up your Python environment and install necessary dependencies.

3. **Configure your settings**:
   - Update `config.py` with your Twilio account details for SMS notifications.
   - Adjust any other settings as needed for your specific use case.

4. **Run the application**: Execute the command provided in the installation section to start the real-time monitoring and notification system.

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit pull requests with improvements, bug fixes, or new features. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [Roboflow Supervision](https://roboflow.com)
- [Twilio](https://www.twilio.com)
- [win10toast](https://github.com/jithurjacob/Windows-10-Toast-Notifications)

## 📬 Contact

For any inquiries or issues, please reach out to [palaria23@gmail.com].

---
