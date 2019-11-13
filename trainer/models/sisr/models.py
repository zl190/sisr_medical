import tensorflow as tf


class MySRResNet(tf.keras.Model):
  def __init__(self):
    super(MySRResNet, self).__init__(name='MySRResNet')
    
    # k9n64s1
    self.conv1 = tf.keras.layers.Conv2D(64, 9, 1, padding='same')
    self.prelu1 = tf.keras.layers.PReLU(alpha_initializer='zeros')
    
    # B residual blocks
    # conv2_1, k3n64s1
    self.conv21a = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
    self.bn21a = tf.keras.layers.BatchNormalization()
    self.prelu21a = tf.keras.layers.PReLU(alpha_initializer='zeros')
    self.conv21b = tf.keras.layers.Conv2D(64, 3, padding='same')
    self.bn21b = tf.keras.layers.BatchNormalization()
    
    self.x21_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)
    self.bn_x21 = tf.keras.layers.BatchNormalization()

     
    # conv2_2, k3n64s1
    self.conv22a = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
    self.bn22a = tf.keras.layers.BatchNormalization()
    self.prelu22a = tf.keras.layers.PReLU(alpha_initializer='zeros')
    self.conv22b = tf.keras.layers.Conv2D(64, 3, padding='same')
    self.bn22b = tf.keras.layers.BatchNormalization()
    
    self.x22_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)
    self.bn_x22 = tf.keras.layers.BatchNormalization()

    # conv2_3, k3n64s1
    self.conv23a = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
    self.bn23a = tf.keras.layers.BatchNormalization()
    self.prelu23a = tf.keras.layers.PReLU(alpha_initializer='zeros')
    self.conv23b = tf.keras.layers.Conv2D(64, 3, padding='same')
    self.bn23b = tf.keras.layers.BatchNormalization()
    
    self.x23_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)
    self.bn_x23 = tf.keras.layers.BatchNormalization()


    # conv2_4, k3n64s1
    self.conv24a = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
    self.bn24a = tf.keras.layers.BatchNormalization()
    self.prelu24a = tf.keras.layers.PReLU(alpha_initializer='zeros')
    self.conv24b = tf.keras.layers.Conv2D(64, 3, padding='same')
    self.bn24b = tf.keras.layers.BatchNormalization()
    
    self.x24_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)
    self.bn_x24 = tf.keras.layers.BatchNormalization()


    # conv2_5, k3n64s1 -- end of B residual block
    self.conv25a = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
    self.bn25a = tf.keras.layers.BatchNormalization()
    self.prelu25a = tf.keras.layers.PReLU(alpha_initializer='zeros')
    self.conv25b = tf.keras.layers.Conv2D(64, 3, padding='same')
    self.bn25b = tf.keras.layers.BatchNormalization()
    
    self.x25_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)
    self.bn_x25 = tf.keras.layers.BatchNormalization()


    # conv3, k3n64s1
    self.conv3 = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.x3_append = tf.keras.layers.Conv2D(64, 1, 1, padding='same', use_bias=False)
    self.bn_x3 = tf.keras.layers.BatchNormalization()

    # conv4_1, k3n256s1
    self.conv41 = tf.keras.layers.Conv2D(256, 3, 1, padding='same')
    self.prelu41 = tf.keras.layers.PReLU(alpha_initializer='zeros')
    # conv4_2
    self.conv42 = tf.keras.layers.Conv2D(256, 3, 1, padding='same')
    self.prelu42 = tf.keras.layers.PReLU(alpha_initializer='zeros')

    # conv5, k9n3s1
    self.conv5 = tf.keras.layers.Conv2D(3, 9, 1, padding='same')

    # self.avgpool = tf.keras.layers.AvgPool2D(3, 2)
    # self.flatten = tf.keras.layers.Flatten()
    # self.dense = tf.keras.layers.Dense(37, activation='softmax')

    

  def call(self, input_tensor, training=False):
    x1 = self.conv1(input_tensor)
    x1 = self.prelu1(x1)
    
     # B residual blocks
    # conv2_1, k3n64s1
    x = self.conv21a(x1)
    x = self.bn21a(x)
    x = self.prelu21a(x)                          
    x = self.conv21b(x)                         
    x = self.bn21b(x)                 
    x_append = self.x21_append(x1)
    x += x_append                   
    x21 = self.bn_x21(x)                 

     
    # conv2_2, k3n64s1
    x = self.conv22a(x21)          
    x = self.bn22a(x)               
    x = self.prelu22a(x)                 
    x = self.conv22b(x)          
    x = self.bn22b(x)                  
    x_append = self.x22_append(x21)
    x += x_append                   
    x22 = self.bn_x22(x)                 

    # conv2_3, k3n64s1
    x = self.conv23a(x22)         
    x = self.bn23a(x)                
    x = self.prelu23a(x)              
    x = self.conv23b(x)          
    x = self.bn23b(x)                
    
    x_append = self.x23_append(x22)
    x += x_append           
    x23 = self.bn_x23(x)                


    # conv2_4, k3n64s1
    x = self.conv24a(x23)           
    x = self.bn24a(x)                
    x = self.prelu24a(x)             
    x = self.conv24b(x)          
    x = self.bn24b(x)                
    
    x_append = self.x24_append(x23)
    x += x_append              
    x24 = self.bn_x24(x)                


    # conv2_5, k3n64s1 -- end of B residual block
    x = self.conv25a(x24)           
    x = self.bn25a(x)                
    x = self.prelu25a(x)             
    x = self.conv25b(x)          
    x = self.bn25b(x)                
    
    x_append = self.x25_append(x24)
    x += x_append              
    x25 = self.bn_x25(x)                


    # conv3, k3n64s1
    x = self.conv3(x25)        
    x = self.bn3(x)                
    x_append = self.x3_append(x1)
    x += x_append
    x = self.bn_x3(x)      


    # conv4_1, k3n256s1
    x = self.conv41(x) 
    x = tf.nn.depth_to_space(x, block_size=2)               
    x = self.prelu41(x)             
    # conv4_2
    x = self.conv42(x)
    x = tf.nn.depth_to_space(x, block_size=2)               
    x = self.prelu42(x)             

    # conv5, k9n3s1
    x = self.conv5(x)         
    return x
  
  
class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__(name='Discriminator')
    
    # k3n64s1
    self.conv1 = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
    self.leakyrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
    
    #  blocks
    # conv2_1, k3n64s2
    self.conv21 = tf.keras.layers.Conv2D(64, 3, 2, padding='same')
    self.bn21 = tf.keras.layers.BatchNormalization()
    self.leakyrelu21 = tf.keras.layers.LeakyReLU(alpha=0.2)
     
    # conv2_2, k3n128s1
    self.conv22 = tf.keras.layers.Conv2D(128, 3, 1, padding='same')
    self.bn22 = tf.keras.layers.BatchNormalization()
    self.leakyrelu22 = tf.keras.layers.LeakyReLU(alpha=0.2)

    # conv2_3, k3n128s2
    self.conv23 = tf.keras.layers.Conv2D(128, 3, 2, padding='same')
    self.bn23 = tf.keras.layers.BatchNormalization()
    self.leakyrelu23 = tf.keras.layers.LeakyReLU(alpha=0.2)

    # conv2_4, k3n256s1
    self.conv24 = tf.keras.layers.Conv2D(256, 3, 1, padding='same')
    self.bn24 = tf.keras.layers.BatchNormalization()
    self.leakyrelu24 = tf.keras.layers.LeakyReLU(alpha=0.2)

    # conv2_5, k3n256s2
    self.conv25 = tf.keras.layers.Conv2D(256, 3, 2, padding='same')
    self.bn25 = tf.keras.layers.BatchNormalization()
    self.leakyrelu25 = tf.keras.layers.LeakyReLU(alpha=0.2)

    # conv2_6, k3n512s1
    self.conv26 = tf.keras.layers.Conv2D(512, 3, 1, padding='same')
    self.bn26 = tf.keras.layers.BatchNormalization()
    self.leakyrelu26 = tf.keras.layers.LeakyReLU(alpha=0.2)

    # conv2_7, k3n512s2 -- end of B residual block
    self.conv27 = tf.keras.layers.Conv2D(512, 3, 2, padding='same')
    self.bn27 = tf.keras.layers.BatchNormalization()
    self.leakyrelu27 = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(1024, activation='softmax')
    self.leakyrelu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
    self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
    

  def call(self, input_tensor, training=False):
    x1 = self.conv1(input_tensor)
    x1 = self.leakyrelu1(x1)
    
    # conv2_1, k3n64s2
    x = self.conv21(x1)
    x = self.bn21(x)
    x = self.leakyrelu21(x)                          
     
    # conv2_2, k3n128s1
    x = self.conv22(x)          
    x = self.bn22(x)               
    x = self.leakyrelu22(x)                        

    # conv2_3, k3n128s2
    x = self.conv23(x)         
    x = self.bn23(x)                
    x = self.leakyrelu23(x)                         

    # conv2_4, k3n256s1
    x = self.conv24(x)           
    x = self.bn24(x)                
    x = self.leakyrelu24(x)            

    # conv2_5, k3n256s2
    x = self.conv25(x)           
    x = self.bn25(x)                
    x = self.leakyrelu25(x)            

    # conv2_6, k3n512s1
    x = self.conv25(x)           
    x = self.bn25(x)                
    x = self.leakyrelu25(x)            

    # conv2_7, k3n512s2 -- end of B residual block
    x = self.conv25(x)           
    x = self.bn25(x)                
    x = self.leakyrelu25(x)            

    x = self.flatten(x)
    x = self.dense1(x)
    x = self.leakyrelu3(x)
    x = self.dense2(x)

    return x