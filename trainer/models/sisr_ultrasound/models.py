import tensorflow as tf
import math

#@title SISR Network
class MySRResNet():
  def __init__(self, shape=(None, None, 1), upsampling_rate=4):
    self.shape = shape
    self.upsampling_rate = upsampling_rate
    
  def __call__(self):
    input_tensor = tf.keras.layers.Input(shape=self.shape)
    x1 = tf.keras.layers.Conv2D(64, 9, padding='same',name='conv1')(input_tensor)
    x1 = tf.keras.layers.PReLU(alpha_initializer='zeros')(x1)
    
     # B residual blocks
    # conv2_1, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_1a')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)                          
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_1b')(x)                         
    x = tf.keras.layers.BatchNormalization()(x)                 
    x21 = x + x1                   

     
    # conv2_2, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_2a')(x21)          
    x = tf.keras.layers.BatchNormalization()(x)               
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)                 
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_2b')(x)          
    x = tf.keras.layers.BatchNormalization()(x)                  
    x22 = x + x21                 

    # conv2_3, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_3a')(x22)         
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)              
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_3b')(x)          
    x = tf.keras.layers.BatchNormalization()(x)                    
    x23 = x + x22           

    # conv2_4, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_4a')(x23)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)             
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_4b')(x)          
    x = tf.keras.layers.BatchNormalization()(x)                    
    x24 = x + x23              

    # conv2_5, k3n64s1 -- end of B residual block
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_5a')(x24)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)             
    x = tf.keras.layers.Conv2D(64, 3, padding='same', name='conv2_5b')(x)          
    x = tf.keras.layers.BatchNormalization()(x)                   
    x25 = x + x24              

    # conv3, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same', name='conv3')(x25)        
    x = tf.keras.layers.BatchNormalization()(x)                
    x = x + x1

    for i in range(int(math.log(self.upsampling_rate,2))):
      # conv4_1, k3n256s1
      x = tf.keras.layers.Conv2D(256, 3, strides=(2,1), padding='same', name='conv4_{}'.format(i+1))(x) 
      x = tf.nn.depth_to_space(x, block_size=2)               
      x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)

    # conv5, k9n3s1
    x = tf.keras.layers.Conv2D(1, 9, padding='same', name='conv5')(x)
    
    return tf.keras.Model(inputs=input_tensor, outputs=x)


class Discriminator():
  def __init__(self, shape=(None, None, 3)):
    self.shape = shape
    

  def __call__(self):
    input_tensor = tf.keras.layers.Input(shape=self.shape) 
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(input_tensor)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # conv2_1, k3n64s2
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)                          
     
    # conv2_2, k3n128s1
    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)          
    x = tf.keras.layers.BatchNormalization()(x)               
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)                        

    # conv2_3, k3n128s2
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)         
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)                         

    # conv2_4, k3n256s1
    x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)            

    # conv2_5, k3n256s2
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)            

    # conv2_6, k3n512s1
    x = tf.keras.layers.Conv2D(512, 3, padding='same')(x)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)            

    # conv2_7, k3n512s2
    x = tf.keras.layers.Conv2D(512, 3, 2, padding='same')(x)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)            

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=input_tensor, outputs=x)
