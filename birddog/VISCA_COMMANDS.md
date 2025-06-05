# BirdDog X1 VISCA Commands Reference

This document lists the VISCA commands implemented in the BirdDog X1 camera control system, based on the official command reference.

## Pan/Tilt/Zoom Commands

### Pan/Tilt Movement
- **Command**: `81 01 06 01 VV WW 0P 0T FF`
- **Description**: Continuous pan/tilt movement
- **Parameters**: 
  - VV: Pan velocity (01-18 hex, 1-24 decimal)
  - WW: Tilt velocity (01-14 hex, 1-20 decimal)  
  - P: Pan direction (01=Right, 02=Left, 03=Stop)
  - T: Tilt direction (01=Up, 02=Down, 03=Stop)

### Zoom Control
- **Zoom In**: `81 01 04 07 2p FF` (p: 0-7 speed)
- **Zoom Out**: `81 01 04 07 3p FF` (p: 0-7 speed)
- **Zoom Stop**: `81 01 04 07 00 FF`

### Position Commands
- **Home Position**: `81 01 06 04 FF`
- **Reset Camera**: `81 01 06 05 FF`

## Camera Settings Commands

### Exposure/Focus/Iris
- **Auto Exposure**: `81 01 04 39 00 FF`
- **Auto Focus On**: `81 01 04 38 02 FF`
- **Auto Focus Off**: `81 01 04 38 03 FF`
- **Auto Iris On**: `81 01 04 0B 00 FF`
- **Auto Iris Off**: `81 01 04 0B 03 FF`

### White Balance
- **Auto**: `81 01 04 35 00 FF`
- **Indoor**: `81 01 04 35 01 FF`
- **Outdoor**: `81 01 04 35 02 FF`
- **Manual**: `81 01 04 35 05 FF`

### Image Settings
- **Backlight On**: `81 01 04 33 02 FF`
- **Backlight Off**: `81 01 04 33 03 FF`
- **Picture Flip On**: `81 01 04 66 02 FF`
- **Picture Flip Off**: `81 01 04 66 03 FF`
- **Gain Up**: `81 01 04 4C 02 FF`
- **Gain Down**: `81 01 04 4C 03 FF`

### Preset Management
- **Set Preset**: `81 01 04 3F 01 pp FF` (pp: preset number 00-FF)
- **Call Preset**: `81 01 04 3F 02 pp FF` (pp: preset number 00-FF)
- **Reset Preset**: `81 01 04 3F 00 pp FF` (pp: preset number 00-FF)

## Command Format

All VISCA commands are sent over UDP to port 52381 with the BirdDog header:
```
[01 00 00 LL 00 00 00 00] + [VISCA_COMMAND]
```
Where LL is the length of the VISCA command + 1.

## Implementation Notes

1. **Speed Ranges**: 
   - Pan: 1-24 (hex: 01-18)
   - Tilt: 1-20 (hex: 01-14)
   - Zoom: 0-7 (hex: 00-07)

2. **Direction Encoding**:
   - Pan: 01=Right, 02=Left, 03=Stop
   - Tilt: 01=Up, 02=Down, 03=Stop

3. **Position Encoding**: 
   - Absolute positions use 16-bit values split into 4-bit nibbles
   - Pan range: -170° to +170° (approx)
   - Tilt range: -30° to +90° (approx)

4. **Preset Numbers**: 
   - Valid range: 0-255 (00-FF hex)
   - GUI limits to 1-10 for user convenience

## GUI Implementation

The updated GUI provides:
- **Pan/Tilt Controls**: Arrow buttons with adjustable speed (1-24)
- **Zoom Controls**: In/Out buttons with adjustable speed (1-7)
- **Camera Settings**: Checkboxes and dropdowns for camera parameters
- **Preset Management**: Set/Call presets 1-10
- **Status Display**: Real-time feedback on command execution

## Error Handling

All commands include try/catch blocks with status updates displayed in the GUI. Network errors, invalid parameters, and camera communication issues are logged to the status panel. 