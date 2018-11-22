#### Za nardit:
##### OpenCV:

##### ROS Koordinatni sistemi:
* transformacija med koordinatnim sistemom kamere in koordinatnim sistemom robota - večji problem bo verjetno orientacija. Cilj je iz točke, ki jo vidi kamera v svojem k.s. dobiti točko v robotovem k.s. - za začetek naj bo z-koordinata točk fiksna (sredina valja)

##### ROS Moveit!:
* ko imamo štartne in ciljne točke v robotovem k.s. lahko z moveit premaknemo robota iz ene točke v drugo. Tukaj nastavim pogoje, da pri pobiranju gripper stisne, pri odlaganju spusti.

#### Narejeno:
##### Povezava, driverji:
* kamera povezana z usb_cam driverjem - komanda za kamera node: 
source camera_ws/devel/setup.bash
rosrun usb_cam usb_cam_node
##### OpenCV:
* delujoča pretvorba iz ROS slik v OpenCV slike z uporabo cv_bridge, urejen CMakeLists in package.xml - referenca http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages (trenutno nariše krog na fiksnih koordinatah žive slike s kamere), za pogon opencv noda naslednji komandi: 
source camera_to_cv/devel/setup.bash
rosrun camera_to_cv camera_to_cv_node 
* zaznavanje posameznih valjev (center, radij) s thresholdingom + zaznavanje oblike kroga
* zazanavanje mej mize s Canny edge detectorjem in shape detectorjem
* zaznavanje lukenj (center, radij) z zaznavanjem oblike kroga
