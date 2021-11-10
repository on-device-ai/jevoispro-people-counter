# JeVois-Pro People Counter
Refer to the [OpenCV People Counter](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/) article to implement a people counter on the [JeVois-Pro Deep Learning Smart Camera](https://www.jevoisinc.com/products/jevois-pro-deep-learning-smart-camera).  
  
First, create a new module on JeVois-Pro GUI:  
![211110_1](https://user-images.githubusercontent.com/44540872/141058205-b5e3a779-b866-4952-bcb4-31b89039674d.png)    
Set the content and create the module: 
![211110_2](https://user-images.githubusercontent.com/44540872/141058282-c7636125-e27d-4d97-ba31-849da1f81f07.png)  
After completion, you can see the newly added module on the JeVois-Pro GUI:  
![211110_3](https://user-images.githubusercontent.com/44540872/141058403-f9f99754-5980-4529-949e-0096a651cac4.png)  
After switching to Ubuntu graphical (Ubuntu Desktop), you can see the newly added PeopleCounter module in the /jevoispro/modules/OnDeviceAI directory:  
![211110_4](https://user-images.githubusercontent.com/44540872/141058502-ba35f6f3-bc4e-40e1-a457-9292504dd1f1.png)  
Then you need to connect to the Internet. I don’t know why JeVois-Pro cannot detect my USB Ethernet Adapter, so I need to install the [device driver](https://github.com/on-device-ai/jevoispro-people-counter/tree/main/r8152-2.15.0). After connecting to the Internet, use the following command to install the dlib package:  
`pip3 install dlib`  
After modifying the [source code](https://github.com/on-device-ai/jevoispro-people-counter/blob/main/OnDeviceAI/PeopleCounter/PeopleCounter.py) of the PeopleCounter module, you can see the execution results as follows:  
![211110](https://user-images.githubusercontent.com/44540872/141058615-fad758b7-4055-4d36-bccd-d187dc138f9d.gif)  