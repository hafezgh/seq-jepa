import blenderproc as bproc
import bpy
import numpy as np
import os
import time
import multiprocessing
import sys
from multiprocessing import Process, Queue

def load_shapenet_object(shapenet_path, synset, obj_id, result_queue, timeout_duration):
    """Function to run in separate process with hard timeout"""
    try:
        # Initialize blenderproc in the subprocess
        bproc.init()
        
        print(f"Starting load of {synset}/{obj_id}...")
        start_time = time.time()
        
        model_obj = bproc.loader.load_shapenet(shapenet_path, used_synset_id=synset, used_source_id=obj_id)
        
        load_time = time.time() - start_time
        
        # Get object info
        bb = model_obj.get_bound_box()
        vertex_count = 0
        face_count = 0
        
        if hasattr(model_obj.blender_obj, 'data') and hasattr(model_obj.blender_obj.data, 'vertices'):
            vertex_count = len(model_obj.blender_obj.data.vertices)
            face_count = len(model_obj.blender_obj.data.polygons)
        
        result = {
            'success': True,
            'load_time': load_time,
            'object_name': model_obj.get_name(),
            'bounding_box': bb,
            'vertex_count': vertex_count,
            'face_count': face_count
        }
        
        result_queue.put(result)
        
    except Exception as e:
        result_queue.put({
            'success': False,
            'error': str(e),
            'load_time': time.time() - start_time if 'start_time' in locals() else 0
        })

# Main analysis code
if __name__ == "__main__":
    # Set your paths
    shapenet_path = "/scratch/users/hafezgh/shapenet"
    val_images_path = "/home/hafezgh/seq-jepa-dev/data/3DIEBench/val_images.npy"
    
    # Load the items list
    items = np.load(val_images_path)
    
    # Specify the problematic object index
    problematic_index = 7550
    
    item = items[problematic_index].strip().strip("/")
    synset = item.split("/")[0]
    obj_id = item.split("/")[1]
    
    print(f"Analyzing object: {synset}/{obj_id}")
    print(f"Full path would be: {shapenet_path}/{synset}/{obj_id}")
    
    # Check if the file exists
    obj_path = f"{shapenet_path}/{synset}/{obj_id}"
    if os.path.exists(obj_path):
        print(f"✓ Object directory exists: {obj_path}")
        
        # List files in the directory
        files = os.listdir(obj_path)
        print(f"Files in directory: {files}")
        
        # Check file sizes
        for file in files:
            file_path = os.path.join(obj_path, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file}: {size_mb:.2f} MB")
                
        # Check models subdirectory if it exists
        models_path = os.path.join(obj_path, "models")
        if os.path.exists(models_path):
            model_files = os.listdir(models_path)
            print(f"Model files: {model_files}")
            for file in model_files:
                file_path = os.path.join(models_path, file)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  models/{file}: {size_mb:.2f} MB")
    else:
        print(f"✗ Object directory does not exist: {obj_path}")
        sys.exit(1)
    
    # Try loading with different timeout values using multiprocessing
    timeout_values = [10]
    
    for timeout_duration in timeout_values:
        print(f"\n--- Attempting to load with {timeout_duration}s timeout (multiprocessing) ---")
        
        # Create a queue to get results from subprocess
        result_queue = Queue()
        
        # Create and start the process
        process = Process(target=load_shapenet_object, 
                         args=(shapenet_path, synset, obj_id, result_queue, timeout_duration))
        
        start_time = time.time()
        process.start()
        
        # Wait for the process to complete or timeout
        process.join(timeout=timeout_duration)
        
        elapsed = time.time() - start_time
        
        if process.is_alive():
            print(f"✗ Process timed out after {elapsed:.2f} seconds - forcefully terminating")
            process.terminate()
            process.join(timeout=5)  # Wait up to 5 seconds for graceful termination
            
            if process.is_alive():
                print("⚠️  Process didn't terminate gracefully, killing it")
                process.kill()
                process.join()
            
            print(f"🔥 This object definitely causes hanging - problematic for batch processing")
            
        else:
            # Process completed, check results
            if not result_queue.empty():
                result = result_queue.get()
                
                if result['success']:
                    print(f"✓ Successfully loaded in {result['load_time']:.2f} seconds")
                    print(f"Object name: {result['object_name']}")
                    print(f"Bounding box shape: {np.array(result['bounding_box']).shape}")
                    print(f"Mesh complexity: {result['vertex_count']} vertices, {result['face_count']} faces")
                    
                    if result['vertex_count'] > 100000:
                        print("⚠️  WARNING: Very high vertex count - this might cause loading issues")
                    if result['face_count'] > 50000:
                        print("⚠️  WARNING: Very high face count - this might cause loading issues")
                    break
                else:
                    print(f"✗ Failed with error: {result['error']}")
                    print(f"Time before failure: {result['load_time']:.2f} seconds")
            else:
                print(f"✗ Process completed but no result returned - unexpected termination")
    
    print(f"\n--- Analysis complete for {synset}/{obj_id} ---")


# import blenderproc as bproc
# import bpy
# import numpy as np
# import time
# from mathutils import Euler
# from multiprocessing import Process, Queue

# def test_rotation_angles(shapenet_path, synset, obj_id, rotation_angles, result_queue):
#     """Test object with specific rotation angles"""
#     try:
#         bproc.init()
#         start_time = time.time()
        
#         # Load object
#         model_obj = bproc.loader.load_shapenet(shapenet_path, used_synset_id=synset, used_source_id=obj_id)
#         load_time = time.time() - start_time
        
#         # Apply rotation
#         rotation_start = time.time()
#         model_obj.blender_obj.rotation_euler = Euler(rotation_angles)
        
#         # Try to get bounding box after rotation (this often triggers the hang)
#         bb = model_obj.get_bound_box()
#         rotation_time = time.time() - rotation_start
        
#         result_queue.put({
#             'success': True, 
#             'load_time': load_time,
#             'rotation_time': rotation_time,
#             'angles': rotation_angles
#         })
        
#     except Exception as e:
#         result_queue.put({
#             'success': False, 
#             'error': str(e),
#             'angles': rotation_angles
#         })

# def test_rotation_hypothesis():
#     # Your problematic object
#     shapenet_path = "/scratch/users/hafezgh/shapenet"
#     synset = "04090263"
#     obj_id = "6ab8bee75629e98d2810ab2c421a9619"
    
#     # Test different rotation ranges
#     test_cases = [
#         # In-distribution (should work)
#         ("In-dist", np.random.uniform(-np.pi/2, np.pi/2, 3)),
#         ("In-dist 2", np.random.uniform(-np.pi/2, np.pi/2, 3)),
        
#         # OOD ranges (might hang)
#         ("OOD extreme", [np.pi*0.8, np.pi*0.9, np.pi*0.7]),  # High positive
#         ("OOD extreme 2", [-np.pi*0.8, -np.pi*0.9, -np.pi*0.7]),  # High negative
#         ("OOD mixed", [np.pi*0.8, -np.pi*0.9, np.pi*0.6]),
        
#         # Edge cases
#         ("Near π", [np.pi-0.1, np.pi-0.1, np.pi-0.1]),
#         ("Near -π", [-np.pi+0.1, -np.pi+0.1, -np.pi+0.1]),
#     ]
    
#     timeout_seconds = 15
    
#     for test_name, angles in test_cases:
#         print(f"\n--- Testing {test_name}: {np.degrees(angles)} degrees ---")
        
#         result_queue = Queue()
#         process = Process(target=test_rotation_angles, 
#                          args=(shapenet_path, synset, obj_id, angles, result_queue))
        
#         start_time = time.time()
#         process.start()
#         process.join(timeout=timeout_seconds)
        
#         elapsed = time.time() - start_time
        
#         if process.is_alive():
#             print(f"🔥 HANGS at rotation {np.degrees(angles)} degrees after {elapsed:.2f}s")
#             process.terminate()
#             process.join(timeout=2)
#             if process.is_alive():
#                 process.kill()
#                 process.join()
#         else:
#             if not result_queue.empty():
#                 result = result_queue.get()
#                 if result['success']:
#                     print(f"✓ Works fine: load={result['load_time']:.2f}s, rotation={result['rotation_time']:.2f}s")
#                 else:
#                     print(f"✗ Error: {result['error']}")
#             else:
#                 print(f"✗ No result returned")

# if __name__ == "__main__":
#     test_rotation_hypothesis()


# import blenderproc as bproc
# import bpy
# import numpy as np
# import time
# from mathutils import Euler
# from multiprocessing import Process, Queue

# def test_cpu_only(shapenet_path, synset, obj_id, result_queue):
#     """Test object loading with CPU-only rendering"""
#     try:
#         bproc.init()
        
#         # Force CPU-only rendering
#         bpy.context.scene.cycles.device = 'CPU'
#         bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'NONE'
        
#         print("Using CPU-only mode")
        
#         start_time = time.time()
#         model_obj = bproc.loader.load_shapenet(shapenet_path, used_synset_id=synset, used_source_id=obj_id)
#         load_time = time.time() - start_time
        
#         # Apply rotation (this was hanging before)
#         rotation_start = time.time()
#         model_obj.blender_obj.rotation_euler = Euler([0.1, 0.1, 0.1])
#         bb = model_obj.get_bound_box()
#         rotation_time = time.time() - rotation_start
        
#         result_queue.put({
#             'success': True, 
#             'load_time': load_time,
#             'rotation_time': rotation_time,
#             'device': 'CPU'
#         })
        
#     except Exception as e:
#         result_queue.put({
#             'success': False, 
#             'error': str(e),
#             'device': 'CPU'
#         })

# def test_gpu_vs_cpu():
#     # Your problematic object
#     shapenet_path = "/scratch/users/hafezgh/shapenet"
#     synset = "04090263"
#     obj_id = "6ab8bee75629e98d2810ab2c421a9619"
    
#     timeout_seconds = 30
    
#     print("=== Testing CPU-only mode ===")
#     result_queue = Queue()
#     process = Process(target=test_cpu_only, 
#                      args=(shapenet_path, synset, obj_id, result_queue))
    
#     start_time = time.time()
#     process.start()
#     process.join(timeout=timeout_seconds)
    
#     elapsed = time.time() - start_time
    
#     if process.is_alive():
#         print(f"🔥 STILL HANGS with CPU-only after {elapsed:.2f}s")
#         process.terminate()
#         process.join(timeout=2)
#         if process.is_alive():
#             process.kill()
#             process.join()
#     else:
#         if not result_queue.empty():
#             result = result_queue.get()
#             if result['success']:
#                 print(f"✅ SUCCESS with CPU-only: load={result['load_time']:.2f}s, rotation={result['rotation_time']:.2f}s")
#                 print("🎯 GPU WAS THE PROBLEM!")
#             else:
#                 print(f"❌ Still failed with CPU: {result['error']}")
#         else:
#             print("❌ No result returned")

# if __name__ == "__main__":
#     test_gpu_vs_cpu()