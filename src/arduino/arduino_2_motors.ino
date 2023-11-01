//by Bishal Shrestha
//header...call function library
#include <Servo.h> 

Servo servox;
Servo servoy;

// User input for servo and position
int UserIn[3];    // raw input from serial buffer, 3 bytes
int ini_byt;       // start byte, begin reading input//initial byte
int servo_num;           // which servo to pulse?
int pos;             // servo angle 0-180
int i;               // iterator
int t=0;    //variable to move to according angle
int u=0;
 

// Common servo setup values
int mini_Pul = 0;   // minimum servo position, us (microseconds)
int maxi_Pul = 120;  // maximum servo position, us



void setup() 
{   
  servoy.attach(9, mini_Pul, maxi_Pul); // Attach each Servo object to a digital pin
  servox.attach(10, mini_Pul, maxi_Pul); // Attach each Servo object to a digital pin
  Serial.begin(115200); // Open the serial connection, 9600 baud

  servoy.write(u);
  servox.write(t);
} 



void loop() 
{ 
  if (Serial.available() > 0) // Wait for serial input (min 3 bytes in buffer)
    {                 //Serial.println("Inside");

  
      pos = Serial.read();// Read the first byte
     
//      if (ini_byt == 58) // If it's really the startbyte (25
//                UserIn[i] = Serial.read();
//              }
//          servo_num = UserIn[0];// First byte = servo to move?          
//          pos = UserIn[1];// Second byte = which position?          
//          if (pos == 255) // Packet error checking and recovery
//              { 
//                servo_num = 255; 
//                Serial.println("Moving0");
//              }
    // ----------X-axis--------------
          if (pos==49 || pos==1 & mini_Pul<t<maxi_Pul)//1=49
            {
              t=t+10;
              servox.write(t); 
            }
         else if (pos==50 & mini_Pul<t<maxi_Pul)//2=50
           {
             t=t-10;
             
             servox.write(t);
           }  
           else if (pos==48 & mini_Pul<t<maxi_Pul) //0
          {
            servox.write(t);
          }
      // ----------Y-axis--------------
           else if (pos==51 & mini_Pul<t<maxi_Pul)//3=51
           {
             u=u-10;
             servoy.write(u);
           } 
           else if (pos==52 & mini_Pul<t<maxi_Pul)//4=52
           {
             u=u+10;
             servoy.write(u);
           } 
           else if (pos==53 & mini_Pul<t<maxi_Pul)//5=53
           {
             servoy.write(u);
           } 
           
         
           Serial.flush();
//        }
    }
}