# realsense_bridge — Intel RealSense D435i szín+mélység adat kiadása ROS1
# Noetic-on keresztül (a robot natúr rendszerén már megvan a librealsense2
# könyvtár + a ROS1 realsense2_camera csomag driverje ehhez a pontos ARM64
# platformhoz, de a Python binding hiányzik és nincs internet mindig —
# ezért a driver-réteget itt, Docker-izoláltan, friss apt-telepítéssel
# hozzuk fel, a natúr rendszert egyáltalán nem érintve).
#
# A konténer:
#  1) elindítja a realsense2_camera ROS1 node-ot (szín + mélység + pontfelhő
#     topicok, TISZTA szenzorolvasás, sosem küld semmit a robotnak)
#  2) elindítja a rosbridge_websocket-et (:9091), hogy a web_dashboard
#     (roslibpy-vel, ugyanaz a minta, mint a SLAM/rosbridge bridge-nél)
#     ki tudja olvasni ezeket a topicokat.
#
# Ld. docs/00-BIZTONSAGI-SZABALYOK.md — a natúr /opt/ros/noetic-t NEM
# módosítjuk, ez egy teljesen külön, konténerben élő ROS1 telepítés.

FROM ros:noetic-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-realsense2-camera \
    ros-noetic-rosbridge-suite \
    ros-noetic-compressed-image-transport \
    ros-noetic-image-transport \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 9091

# A realsense2_camera node image_transport-tal maga is kiadja a /compressed
# topicokat — külön republish nem kell. A depth /compressed ALAPBÓL 8-bites
# JPEG (mélységre használhatatlan), ezért a roslaunch ELŐTT rosparam-mal
# PNG-re (16-bit, veszteségmentes) állítjuk; a dynamic_reconfigure szerver
# induláskor ezt veszi át, a kamera-node újraindításai után is.
# align_depth:=true kell a mission_control/perception pillérnek (YOLO bbox
# -> 3D pozíció, a mélységkép pixelre pontosan a színképhez igazítva).
#
# FONTOS: a ROS1 rosbridge_websocket egy futó roscore-hoz (master, :11311)
# próbál csatlakozni indításkor — ha nincs kifejezetten elindítva előtte,
# "Failed to contact master, Retrying..." hibával örökre retry-el, sosem
# nyitja meg a websocket portot. Ezért itt explicit ELŐSZÖR a roscore fut.
#
# A kamera-node egy újraindító ciklusban fut: 2026-09-19-én a tartós
# konténerben a node csendben elhalt (USB "uvc streamer watchdog"), a
# konténer és a rosbridge viszont "Up" maradt — kép nélkül.
CMD ["/bin/bash", "-c", "\
    source /opt/ros/noetic/setup.bash && \
    roscore & \
    sleep 4 && \
    rosparam set /camera/aligned_depth_to_color/image_raw/compressed/format png && \
    rosparam set /camera/aligned_depth_to_color/image_raw/compressed/png_level 1 && \
    ( while true; do \
        roslaunch realsense2_camera rs_camera.launch \
          align_depth:=true enable_pointcloud:=false \
          color_width:=640 color_height:=480 color_fps:=15 \
          depth_width:=640 depth_height:=480 depth_fps:=15 \
          enable_infra1:=false enable_infra2:=false \
          enable_gyro:=false enable_accel:=false; \
        echo 'realsense2_camera exited, restarting in 3 s'; sleep 3; \
      done ) & \
    sleep 8 && \
    roslaunch rosbridge_server rosbridge_websocket.launch port:=9091 & \
    wait \
"]
