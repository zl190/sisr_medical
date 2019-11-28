import tensorflow as tf


class MySRResNet():
  def __init__(self, shape=(None, None, 3)):
    self.shape = shape
    

  def __call__(self):
    input_tensor = tf.keras.layers.Input(shape=self.shape)
    x1 = tf.keras.layers.Conv2D(64, 9, 1, padding='same')(input_tensor)
    x1 = tf.keras.layers.PReLU(alpha_initializer='zeros')(x1)
    
     # B residual blocks
    # conv2_1, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)                          
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)                         
    x = tf.keras.layers.BatchNormalization()(x)                 
    x_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)(x1)
    x21 = x + x_append                   
    #x21 = tf.keras.layers.BatchNormalization()(x)                 

     
    # conv2_2, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x21)          
    x = tf.keras.layers.BatchNormalization()(x)               
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)                 
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)          
    x = tf.keras.layers.BatchNormalization()(x)                  
    x_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)(x21)
    x22 = x + x_append                   
    #x22 = tf.keras.layers.BatchNormalization()(x)                 

    # conv2_3, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x22)         
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)              
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)          
    x = tf.keras.layers.BatchNormalization()(x)                
    
    x_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)(x22)
    x23 = x + x_append           
    #x23 = tf.keras.layers.BatchNormalization()(x)                


    # conv2_4, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x23)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)             
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)          
    x = tf.keras.layers.BatchNormalization()(x)                
    
    x_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)(x23)
    x24 = x + x_append              
    #x24 = tf.keras.layers.BatchNormalization()(x)                


    # conv2_5, k3n64s1 -- end of B residual block
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x24)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)             
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)          
    x = tf.keras.layers.BatchNormalization()(x)                
    
    x_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)(x24)
    x25 = x + x_append              
    #x25 = tf.keras.layers.BatchNormalization()(x)                


    # conv3, k3n64s1
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x25)        
    x = tf.keras.layers.BatchNormalization()(x)                
    x_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)(x1)
    x = x + x_append
    #x = tf.keras.layers.BatchNormalization()(x)      


    # conv4_1, k3n256s1
    x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(x) 
    x = tf.nn.depth_to_space(x, block_size=2)               
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)             
    # conv4_2
    x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(x)
    x = tf.nn.depth_to_space(x, block_size=2)               
    x = tf.keras.layers.PReLU(alpha_initializer='zeros')(x)             

    # conv5, k9n3s1
    x = tf.keras.layers.Conv2D(3, 9, 1, padding='same')(x)
    
    return tf.keras.Model(inputs=input_tensor, outputs=x)
  

class Discriminator():
  def __init__(self, shape=(None, None, 3)):
    self.shape = shape
    

  def __call__(self):
    input_tensor = tf.keras.layers.Input(shape=self.shape) 
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(input_tensor)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # conv2_1, k3n64s2
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)                          
     
    # conv2_2, k3n128s1
    x = tf.keras.layers.Conv2D(128, 3, 1, padding='same')(x)          
    x = tf.keras.layers.BatchNormalization()(x)               
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)                        

    # conv2_3, k3n128s2
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)         
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)                         

    # conv2_4, k3n256s1
    x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(x)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)            

    # conv2_5, k3n256s2
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)           
    x = tf.keras.layers.BatchNormalization()(x)                
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)            

    # conv2_6, k3n512s1
    x = tf.keras.layers.Conv2D(512, 3, 1, padding='same')(x)           
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