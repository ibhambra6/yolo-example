"""
NDI Debug Tool - Helps diagnose NDI connection issues
Lists all available NDI sources on the network and tests camera connectivity.
"""
import socket
import time
import NDIlib as ndi
from birddog import X1Visca

def test_camera_connectivity(ip, port=52381):
    """Test if camera is reachable via VISCA-IP"""
    print(f"\n🔍 Testing camera connectivity to {ip}:{port}")
    try:
        cam = X1Visca(ip, port)
        # Try to send a basic query command
        cam._send(bytes([0x81, 0x09, 0x04, 0x00, 0xFF]))  # CAM_PowerInq
        print(f"✅ Camera VISCA-IP connection successful")
        return True
    except Exception as e:
        print(f"❌ Camera VISCA-IP connection failed: {e}")
        return False

def list_ndi_sources():
    """List all available NDI sources on the network"""
    print("\n🔍 Scanning for NDI sources...")
    
    if not ndi.initialize():
        print("❌ Failed to initialize NDI")
        return []
    
    finder = ndi.find_create_v2()
    if finder is None:
        print("❌ Failed to create NDI finder")
        return []
    
    sources = []
    max_wait_time = 10  # Wait up to 10 seconds
    start_time = time.time()
    
    print("⏳ Waiting for NDI sources (up to 10 seconds)...")
    
    while time.time() - start_time < max_wait_time:
        current_sources = ndi.find_get_current_sources(finder)
        if current_sources:
            sources = current_sources
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    
    print()  # New line after dots
    
    if not sources:
        print("❌ No NDI sources found on the network")
        print("\n💡 Troubleshooting tips:")
        print("   1. Ensure NDI is enabled on your BirdDog X1 camera")
        print("   2. Check that camera and computer are on same network")
        print("   3. Verify camera IP address is correct")
        print("   4. Check firewall settings (NDI uses ports 5353, 5960-5989)")
        return []
    
    print(f"✅ Found {len(sources)} NDI source(s):")
    for i, source in enumerate(sources):
        ndi_name = source.ndi_name.decode() if hasattr(source.ndi_name, 'decode') else str(source.ndi_name)
        url_address = source.url_address.decode() if hasattr(source.url_address, 'decode') else str(source.url_address)
        print(f"   {i+1}. Name: '{ndi_name}'")
        print(f"      URL: {url_address}")
    
    ndi.find_destroy(finder)
    return sources

def test_ndi_connection(source_name="CAM"):
    """Test NDI connection with specific source name"""
    print(f"\n🔍 Testing NDI connection to source containing '{source_name}'...")
    
    if not ndi.initialize():
        print("❌ Failed to initialize NDI")
        return False
    
    finder = ndi.find_create_v2()
    recv = ndi.recv_create_v3()
    
    if finder is None or recv is None:
        print("❌ Failed to create NDI finder or receiver")
        return False
    
    source = None
    max_wait_time = 5
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        sources = ndi.find_get_current_sources(finder)
        for s in sources:
            ndi_name = s.ndi_name.decode() if hasattr(s.ndi_name, 'decode') else str(s.ndi_name)
            if source_name in ndi_name:
                source = s
                break
        if source:
            break
        time.sleep(0.25)
    
    if not source:
        print(f"❌ No NDI source found containing '{source_name}'")
        ndi.find_destroy(finder)
        return False
    
    print(f"✅ Found matching NDI source: {source.ndi_name.decode() if hasattr(source.ndi_name, 'decode') else str(source.ndi_name)}")
    
    try:
        ndi.recv_connect(recv, source)
        print("✅ NDI receiver connected successfully")
        
        # Try to capture a frame
        print("⏳ Testing frame capture...")
        frame_type, video_frame, audio_frame, metadata = ndi.recv_capture_v2(recv, timeout_in_ms=5000)
        if frame_type == ndi.FRAME_TYPE_VIDEO and video_frame.data is not None:
            print(f"✅ Frame captured successfully ({video_frame.xres}x{video_frame.yres})")
            ndi.recv_free_video_v2(recv, video_frame)
            return True
        else:
            print("❌ Failed to capture frame (no data)")
            return False
            
    except Exception as e:
        print(f"❌ NDI connection failed: {e}")
        return False
    finally:
        ndi.find_destroy(finder)

def main():
    """Main diagnostic function"""
    print("🔧 BirdDog X1 NDI Diagnostics")
    print("=" * 40)
    
    # Test camera connectivity
    camera_ip = "192.168.0.13"  # From config
    camera_ok = test_camera_connectivity(camera_ip)
    
    # List all NDI sources
    sources = list_ndi_sources()
    
    # Test specific NDI connection
    ndi_name = "CAM"  # From config
    ndi_ok = test_ndi_connection(ndi_name)
    
    print("\n📊 Summary:")
    print(f"   Camera VISCA-IP: {'✅ OK' if camera_ok else '❌ FAIL'}")
    print(f"   NDI Sources Found: {len(sources)}")
    print(f"   NDI Connection: {'✅ OK' if ndi_ok else '❌ FAIL'}")
    
    if not camera_ok:
        print(f"\n💡 Camera troubleshooting:")
        print(f"   - Check if camera is powered on")
        print(f"   - Verify IP address {camera_ip} is correct")
        print(f"   - Ensure camera and computer are on same network")
        print(f"   - Test with ping: ping {camera_ip}")
    
    if not ndi_ok and sources:
        print(f"\n💡 NDI troubleshooting:")
        print(f"   - Your config expects NDI name containing '{ndi_name}'")
        source_names = []
        for s in sources:
            if hasattr(s.ndi_name, 'decode'):
                source_names.append(s.ndi_name.decode())
            else:
                source_names.append(str(s.ndi_name))
        print(f"   - Available sources: {source_names}")
        print(f"   - Update NDI_NAME in config to match actual source name")

if __name__ == "__main__":
    main() 