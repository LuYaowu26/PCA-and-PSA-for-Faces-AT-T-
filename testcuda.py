import cupy as cp

dev = cp.cuda.Device(0)
print("Device ID:", dev.id)
print("Device Name:", cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode('utf-8'))
print("Compute capability:", cp.cuda.runtime.getDeviceProperties(dev.id)['major'], ".", cp.cuda.runtime.getDeviceProperties(dev.id)['minor'], sep="")
