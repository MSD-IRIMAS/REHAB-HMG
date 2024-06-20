# JOINT_MAP = {
#     'SpineBase':0,
#     'SpineMid':1,
#     'Head':2,
#     'LeftShoulder':3,
#     'LeftElbow':4,
#     'LeftWrist':5,
#     'LeftHand':6,
#     'RightShoulder':7,
#     'RightElbow':8,
#     'RightWrist':9,
#     'RightHand':10,
#     'RightHip':11,
#     'LeftKnee':12,
#     'LeftAnkle':13,
#     'LeftHip':14,
#     'RightKnee':15,
#     'RightAnkle':16,
#     'Neck':17,
    
    
    
#     }
JOINT_MAP = {0: 'SpineBase', 
1: 'SpineMid', 
2: 'Head', 
3: 'LeftShoulder',
 4: 'LeftElbow', 
 5: 'LeftWrist', 
 6: 'LeftHand', 
 7: 'RightShoulder',
  8: 'RightElbow', 
  9: 'RightWrist', 
  10: 'RightHand', 
  11: 'RightHip', 
  12: 'LeftKnee', 
  13: 'LeftAnkle', 
  14: 'LeftHip', 
  15: 'RightKnee', 
  16: 'RightAnkle', 
  17: 'Neck'}

joint_pairs = [
    (3,4),(4,5),(5,6),(12,13),(11,12),(0,11),(0,14),(14,15),(15,16),(7,8),(8,9),(9,10),(17,7),(1,0),(3,17),(2,17),(1,17)
    ]