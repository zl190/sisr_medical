import tensorflow as tf


def get_central_mask2d(size=tf.constant((128, 128), dtype=tf.int32), template_shape=tf.constant((256, 256), dtype=tf.int32)):
    """
        size is a 2d tuple or tensor
        template_shape is a 2d tuple or tensor
        returns a central cropped mask and a bbox compatible with tf.image.crop_to_bounding_box
    """    
    start = tf.cast((template_shape-size)/2, dtype=tf.int32)
    end = tf.cast(size+start, dtype=tf.int32)
    
    height = tf.range(start[0], end[0], delta=1, dtype=tf.int32)
    width = tf.range(start[1], end[1], delta=1, dtype=tf.int32)
    
    bbox = {
        'offset_height': start[0],
        'offset_width': start[1],
        'target_height': size[0],
        'target_width': size[1]        
    }
    return get_mask(height, width, template_shape=template_shape), bbox


def get_mask(*args, template_shape=tf.constant((256, 256), dtype=tf.int32)):
    """
        args is a n-list 1D tensors
        returns a ND mask of ones
    """
    
    indices = tf.stack(tf.meshgrid(*args), axis=-1)
    indices = tf.reshape(indices, [-1, tf.shape(indices)[-1]])
    updates = tf.ones(tf.shape(indices)[0], dtype=tf.int64)
    mask = tf.scatter_nd(indices, updates, template_shape)
    mask = tf.expand_dims(mask, -1)
    return mask


def _pad_bbox_to_size(bbox, size):
    """pad a bbox to target size, no care now about whether the postion is valid
    
      Params:
        bbox: [xmin, ymin, xmax, ymax]
        size: tf.int32
      
      return:
        new_bbox
    """
    offset_height, offset_width, target_height, target_width = _convert_bbox_to_offset(bbox)
    residue_height = size - target_height
    residue_width = size - target_width
    
    xmin = tf.cast(tf.math.round(bbox[0]), dtype=tf.int32)
    ymin = tf.cast(tf.math.round(bbox[1]), dtype=tf.int32)
    xmax = tf.cast(tf.math.round(bbox[2]), dtype=tf.int32)
    ymax = tf.cast(tf.math.round(bbox[3]), dtype=tf.int32)
    
    divisor = 2
    residue = 1
    new_ymin = ymin - tf.math.floordiv(residue_height,divisor)
    new_xmin = xmin - tf.math.floordiv(residue_width,divisor)
    new_ymax = ymax + tf.cond(tf.equal(tf.math.floordiv(residue_height,divisor)*divisor, residue_height), 
                              lambda: tf.math.floordiv(residue_height,divisor), 
                              lambda: tf.math.floordiv(residue_height,divisor) + residue)
    new_xmax = xmax + tf.cond(tf.equal(tf.math.floordiv(residue_width,divisor)*divisor, residue_width), 
                              lambda: tf.math.floordiv(residue_width,divisor), 
                              lambda: tf.math.floordiv(residue_width,divisor) + residue)
    return [new_xmin, new_ymin, new_xmax, new_ymax]
  

def _shift_bbox_into_image(bbox, image_shape):
    """keep the size of bbox but shift it into an image, make the postion valid
    
      Assert: bbox should be smaller than the image
      [shift_right] if xmin < 0 and xmax < image_xmax: 
                        new_xmin = 0 
                        new_xmax = xmax - xmin 
      [shift_left] if xmin > 0 and xmax > image_xmax: 
                        new_xmin = xim - (xmax - image_xmax)
                        new_xmax = image_xmax
      [shift_down] if ymin < 0 and ymax < image_ymax: 
                        new_ymin = 0
                        new_ymax = ymax - ymin 
      [shift_up] if ymin > 0 and ymax > image_ymax: 
                        new_ymin = ymax - image_ymax
                        new_ymax = image_ymax
    """
    image_ymin = tf.cast(0, dtype=tf.int32)
    image_xmin = tf.cast(0, dtype=tf.int32)
    image_ymax = tf.cast(image_shape[0], dtype=tf.int32)
    image_xmax = tf.cast(image_shape[1], dtype=tf.int32)
    
    xmin = tf.cast(tf.math.round(bbox[0]), dtype=tf.int32)
    ymin = tf.cast(tf.math.round(bbox[1]), dtype=tf.int32)
    xmax = tf.cast(tf.math.round(bbox[2]), dtype=tf.int32)
    ymax = tf.cast(tf.math.round(bbox[3]), dtype=tf.int32)
    
    shift_right = tf.math.logical_and(tf.less(xmin, 0), tf.less(xmax, image_xmax))
    shift_left = tf.math.logical_and(tf.greater(xmin, 0), tf.greater(xmax, image_xmax))
    shift_down = tf.math.logical_and(tf.less(ymin,0), tf.less(ymax, image_ymax))
    shift_up = tf.math.logical_and(tf.greater(ymin,0), tf.greater(ymax, image_ymax))
      
    # shift_right
    new_xmin = tf.cond(shift_right, lambda: 0, lambda: xmin)
    new_xmax = tf.cond(shift_right, lambda: xmax-xmin, lambda: xmax)
    
    # shift_left
    new_xmin = tf.cond(shift_left, lambda: xmin - (xmax - image_xmax), lambda: new_xmin)
    new_xmax = tf.cond(shift_left, lambda: image_xmax, lambda: new_xmax)
    
    # shift_down
    new_ymin = tf.cond(shift_down, lambda: 0, lambda: ymin)
    new_ymax = tf.cond(shift_down, lambda: ymax - ymin, lambda: ymax)

    # shift_up
    new_ymin = tf.cond(shift_up, lambda: ymin - (ymax - image_ymax), lambda: new_ymin)
    new_ymax = tf.cond(shift_up, lambda: image_ymax, lambda: new_ymax)
    
    return [new_xmin, new_ymin, new_xmax, new_ymax]

  
def _convert_bbox_to_mask(bbox, image_shape):
    """generate a 2-d binary mask, all zeros except the bbox area
      
      bbox: [xmin, ymin, xmax, ymax]
      image_shape: [height, width, depth], depth will be ignored
      
      return:
        2-d binary mask, zero-initialized
    """
    xmin = tf.cast(tf.math.round(bbox[0]), dtype=tf.int32)
    ymin = tf.cast(tf.math.round(bbox[1]), dtype=tf.int32)
    xmax = tf.cast(tf.math.round(bbox[2]), dtype=tf.int32)
    ymax = tf.cast(tf.math.round(bbox[3]), dtype=tf.int32)
    
    r_range = tf.range(xmin, xmax)
    c_range = tf.range(ymin, ymax)
    i_coords, j_coords = tf.meshgrid(r_range, c_range)
    
    indices = tf.stack([j_coords,i_coords], axis=-1)
    indices = tf.reshape(indices, [tf.size(i_coords),2])
    updates = tf.ones(tf.size(i_coords), dtype=tf.int64)
    shape = [image_shape[0], image_shape[1]]
    mask = tf.scatter_nd(indices, updates, shape)
    return mask
    
    
def _convert_mask_to_bbox(mask):
    """find out the bbox postion inside the mask
    """
    ymin = tf.where(mask)[:,0][0]
    ymax = tf.where(mask)[:,0][-1]
    xmin = tf.where(mask)[:,1][0]
    xmax = tf.where(mask)[:,1][-1]
    return [xmin, ymin, xmax, ymax]


def _convert_bbox_to_offset(bbox):
    """convert the data format
    
      Params:
        bbox: [xmin, ymin, xmax, ymax]
      return:
        offset: [offset_height=ymin, offset_width=xmin, 
                 target_height=ymax - ymin, target_width=xmax - xmin]
      
    """
    xmin = tf.cast(tf.math.round(bbox[0]), dtype=tf.int32)
    ymin = tf.cast(tf.math.round(bbox[1]), dtype=tf.int32)
    xmax = tf.cast(tf.math.round(bbox[2]), dtype=tf.int32)
    ymax = tf.cast(tf.math.round(bbox[3]), dtype=tf.int32)
    
    offset_height = ymin
    offset_width = xmin
    target_height = ymax - ymin
    target_width = xmax - xmin
    
    return [offset_height, offset_width, target_height, target_width]
  

def _convert_offset_to_bbox(offset):
    """convert the data format
    
      Params:
        offset: [offset_height, offset_width, target_height, target_width]
      return:
        bbox: [xmin=offset_width, ymin=offset_height, 
               xmax=target_width + offset_width, ymax=target_height + offset_height]
      
    """
    offset_height = offset[0]
    offset_width = offset[1] 
    target_height = offset[2]
    target_width = offset[3]
    
    ymin = offset_height
    xmin = offset_width
    ymax = target_height + offset_height
    xmax = target_width + offset_width
    
    return [xmin, ymin, xmax, ymax]
  
    
def clip_image(image, mask, size):
    """clip the image and mask according to the given size
    
      clip should be exactually the size, clip should lay in the image, clip 
      should contain the mask, ideally the mask centers inside the clip except 
      the mask really close to the edge. 
      
      image: [height, width, depth] # depth should be 1 or None
      mask: [height, width, depth] # depth should be 1 or None
      size: tf.int32

      return: 
        clipped_com: [height, width, depth=2], clipped image stacked with 
          the clipped mask
    """
    mask = tf.cast(mask, dtype=tf.int32)
    bbox = _convert_mask_to_bbox(tf.squeeze(mask))
    center_sized_bbox = _pad_bbox_to_size(bbox, size)
    center_sized_bbox = _shift_bbox_into_image(center_sized_bbox, tf.shape(image))
    mask = tf.cast(mask, dtype=tf.float32)
    image_mask = tf.concat([image, mask], -1)
    offset = _convert_bbox_to_offset(center_sized_bbox)   
    return tf.image.crop_to_bounding_box(image_mask, *offset)


def batch_clip_image(image, mask, size):
    """only support 4 channels clip, return clipped image and mask
    
      image and mask has the same shape, designed for lesion GANs
    
      Params:
        image: [batch_size, height, width, depth], tf.float64
        mask: [batch_size, height, width, depth], tf.int64
        size: tf.int32, the target size of clip
      
      return:
        clipped_image: [batch_size, clipped_height, clipped_width, depth], tf.float64
        clipped_mask: [batch_size, clipped_height, clipped_width, depth], tf.int64
    """
    # data processing for input
    clipped_com = tf.map_fn(lambda x: clip_image(x[0], x[1], size), (image, mask), dtype=tf.float32)
    
    # data processing for output
    clipped_image = clipped_com[:,:,:,:-1]
    clipped_mask = clipped_com[:,:,:,-1:]
    return clipped_image, clipped_mask
