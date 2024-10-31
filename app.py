#imports
import streamlit as st
import streamlit_authenticator as stauth
from PIL import Image
from ultralytics import YOLO
import os
import json
import hashlib
from datetime import datetime
import pandas as pd
from collections import defaultdict
from fpdf import FPDF
from collections import defaultdict
import importlib
import fitz 
from fpdf.enums import XPos, YPos
import asyncio



# Import forensic tools for different digital evidence types
from digital_evidences.laptop import get_forensic_tools as laptop_tools
from digital_evidences.calculator import get_forensic_tools as calculator_tools
from digital_evidences.camera import get_forensic_tools as camera_tools
from digital_evidences.computer_keyboard import get_forensic_tools as computer_keyboard_tools
from digital_evidences.computer_mouse import get_forensic_tools as computer_mouse_tools
from digital_evidences.digital_clock import get_forensic_tools as digital_clock_tools
from digital_evidences.headphones import get_forensic_tools as headphones_tools
from digital_evidences.ipod import get_forensic_tools as ipod_tools
from digital_evidences.microphone import get_forensic_tools as microphone_tools
from digital_evidences.mobile_phone import get_forensic_tools as mobile_phone_tools
from digital_evidences.musical_keyboard import get_forensic_tools as musical_keyboard_tools
from digital_evidences.smartwatch import get_forensic_tools as smartwatch_tools
from digital_evidences.television import get_forensic_tools as television_tools
from digital_evidences.tripod import get_forensic_tools as tripod_tools
from digital_evidences.vehicle_registration_plate import get_forensic_tools as vehicle_registration_plate_tools
from digital_evidences.watch import get_forensic_tools as watch_tools



            
# Updated function to display forensic tools based on the selected evidence
def display_forensic_tools(evidence_type):
    # Normalize evidence type to match key in FORENSIC_TOOLS dictionary
    evidence_type_lower = evidence_type.lower().strip()

    # Create a mapping of evidence types to their corresponding tools
    tool_mappings = {
        "laptop": "Laptop",
        "computer": "Laptop",
        "mobile phone": "Mobile phone",
        "phone": "Mobile phone",
        "smartphone": "Mobile phone",
        "cellphone": "Mobile phone",
        "smartwatch": "Smartwatch",
        "watch": "Smartwatch",
        "computer keyboard": "Computer Keyboard",
        "digital clock": "Digital Clock",
        "calculator": "Calculator",
        "camera": "Camera",
        "computer mouse": "Computer Mouse",
        "headphones": "Headphones",
        "ipod": "ipod",
        "microphone": "Microphone",
        "musical keyboard": "Musical Keyboard",
        "Tripod": "Tripod",
        "television":"Television",
        "vehicle registration plate": "Vehicle registration plate"
    }
    
    # Retrieve the correctly mapped type, ensuring consistent capitalization
    matched_type = tool_mappings.get(evidence_type_lower, evidence_type_lower.title())

    # Ensure that the matched type is properly capitalized to align with FORENSIC_TOOLS keys
    if matched_type in FORENSIC_TOOLS:
        st.markdown(f"### Forensic Tools for {matched_type}")
        st.markdown("---")
        
        for tool in FORENSIC_TOOLS[matched_type]["tools"]:
            st.markdown(f"**{tool['name']}**")
            st.write(f"{tool['description']}")
            st.markdown("**Use Cases:**")
            for use_case in tool["use_cases"]:
                st.write(f"- {use_case}")
            st.markdown("**How to Use:**")
            st.write(tool["how_to_use"])
            st.markdown("**Key Features:**")
            for feature in tool["features"]:
                st.write(f"- {feature}")
            st.markdown(f"**Where to Find:** [Visit Website]({tool['where_to_find']})")
            st.markdown("---")
    else:
        st.warning(f"No specific tools information available for '{evidence_type}' yet.")




# Updated Display detection results function with non-digital evidence descriptions
# Function to display detection results

def display_detection_results(results, is_history=False, unique_id=None):
    # Function content here

    if results is not None:
        from collections import defaultdict
        digital_evidence = defaultdict(list)
        non_digital_evidence = defaultdict(list)

        # Generate a unique identifier if none is provided
        if unique_id is None:
            unique_id = str(datetime.now().timestamp())

        # Normalize digital evidence items for better comparison
        digital_evidence_set = set(item.lower() for item in DIGITAL_EVIDENCE)

        # Process results based on whether they're from history or live detection
        if is_history:
            for item in results:
                label = item['label'].strip().capitalize()  # Capitalize to match dictionary case
                confidence = item['confidence']

                # Check if label is in digital or non-digital evidence list
                if label.lower() in digital_evidence_set:
                    digital_evidence[label].append(confidence)
                else:
                    non_digital_evidence[label].append(confidence)
        else:
            boxes = results[0].boxes
            if boxes is not None:
                data = boxes.data.cpu().numpy()
                for box in data:
                    label = model.names[int(box[5])].strip().capitalize()  # Capitalize to match dictionary case
                    confidence = float(box[4])

                    # Check if label is in digital or non-digital evidence list
                    if label.lower() in digital_evidence_set:
                        digital_evidence[label].append(confidence)
                    else:
                        non_digital_evidence[label].append(confidence)

        def prepare_data(evidence_dict):
            return {
                "Label": list(evidence_dict.keys()),
                "Repetitions": [len(confs) for confs in evidence_dict.values()],
                "Avg Confidence": [round(sum(confs) / len(confs), 2) for confs in evidence_dict.values()]
            }

        digital_df = pd.DataFrame(prepare_data(digital_evidence))
        non_digital_df = pd.DataFrame(prepare_data(non_digital_evidence))

        # Display detected digital evidence with expanders for tools
        st.subheader("Digital Evidence")
        if not digital_df.empty:
            for idx, row in digital_df.iterrows():
                label = row['Label']
                st.markdown(f"**{label}** (Confidence: {row['Avg Confidence']}%) - Repetitions: {row['Repetitions']}")
                
                # Use an expander for viewing tools
                with st.expander(f"🔍 View Tools for {label}"):
                    display_forensic_tools(label)
            st.dataframe(digital_df)
        else:
            st.write("No digital evidence detected.")

        # Display detected non-digital evidence with descriptions
        st.subheader("Non-Digital Evidence")
        if not non_digital_df.empty:
            for idx, row in non_digital_df.iterrows():
                label = row['Label']
                st.markdown(f"**{label}** (Confidence: {row['Avg Confidence']}%) - Repetitions: {row['Repetitions']}")

                # Retrieve and display non-digital evidence description
                description = NON_DIGITAL_EVIDENCE_DESC.get(label, "No specific forensic details available.")
                st.markdown(f"*{description}*")
            st.dataframe(non_digital_df)
        else:
            st.write("No non-digital evidence detected.")
    else:
        st.write("No objects detected.")


# Set page configuration
st.set_page_config(page_title="YOLOv8 Object Detector", layout="wide")

DIGITAL_EVIDENCE = [
    "Laptop", "smartwatch", "Microphone", "Headphones",
    "Computer keyboard", "Mobile phone", "Watch",
    "Digital Clock", "Calculator", "Camera", "Computer Mouse","ipod","Musical Keyboard","Television",
    "Tripod","Vehicle registration plate"
]

# Define categories and their forensic notes
CATEGORY_DESCRIPTIONS = {
    "biological_evidence": "This item can contain biological materials such as DNA or fingerprints, helping identify individuals involved.",
    "trace_evidence": "Residues or fibers on this item can link it to a specific location or person, aiding in establishing connections.",
    "physical_markings": "Marks or damage on this object can provide insights into usage patterns or potential interactions during an incident.",
    "contextual_clues": "This item may offer contextual clues, such as location, time, or usage, that assist in reconstructing the sequence of events.",
}

# Sample non-digital evidence categories
NON_DIGITAL_EVIDENCE_CATEGORIES = {
    "biological_evidence": ["Clothing", "Backpack", "Hat", "Glove", "Scarf", "Towel", "Pillow", "Toothbrush"],
    "trace_evidence": ["Adhesive tape", "Envelope", "Coin", "Bottle", "Cupboard", "Curtain", "Briefcase", "Book"],
    "physical_markings": ["Knife", "Baseball bat", "Axe", "Hammer", "Vehicle", "Chair", "Table", "Wall"],
    "contextual_clues": ["Clock", "Calendar", "Watch", "Telephone", "Lamp", "Staircase", "Street sign", "Traffic light"]
}

# Generate descriptions for all non-digital items
NON_DIGITAL_EVIDENCE_DESC = {}
for category, items in NON_DIGITAL_EVIDENCE_CATEGORIES.items():
    for item in items:
        NON_DIGITAL_EVIDENCE_DESC[item] = CATEGORY_DESCRIPTIONS[category]

# Generic description for unclassified items
generic_description = "This object may hold various forensic clues, such as location traces, biological residues, or physical markings."
OTHER_NON_DIGITAL_ITEMS = ["Accordion", "Aircraft", "Airplane", "Alarm clock", "Apple", "Bed", "Bee", "Bicycle", "Boat", "Bridge"]
for item in OTHER_NON_DIGITAL_ITEMS:
    NON_DIGITAL_EVIDENCE_DESC[item] = generic_description




# Define forensic tools and their details for each digital evidence type
FORENSIC_TOOLS = {
    "Laptop": {
        "tools": [
            {
                "name": "EnCase Forensic",
                "description": "Professional-grade computer forensics software for detailed analysis of hard drives and other storage devices.",
                "use_cases": ["Data Recovery", "File System Analysis", "Registry Analysis", "Email Analysis"],
                "where_to_find": "https://www.opentext.com/products/encase-forensic",
                "how_to_use": "1. Create a forensic image of the laptop\n2. Load the image into EnCase\n3. Use built-in tools for analysis\n4. Generate detailed reports",
                "features": ["Write blocking", "Evidence preservation", "Timeline analysis", "Hash verification"]
            },
            {
                "name": "FTK (Forensic Toolkit)",
                "description": "Comprehensive digital forensics platform for examination of computers.",
                "use_cases": ["Data Carving", "Memory Analysis", "Password Recovery", "Malware Detection"],
                "where_to_find": "https://accessdata.com/products-services/forensic-toolkit-ftk",
                "how_to_use": "1. Create forensic image\n2. Import to FTK\n3. Run automated analysis\n4. Examine results using various filters",
                "features": ["Email analysis", "Registry viewer", "Data visualization", "Keyword searching"]
            },
            {
                "name": "Autopsy",
                "description": "Free, open-source digital forensics platform.",
                "use_cases": ["File Analysis", "Web Artifacts", "Keyword Search", "Timeline Analysis"],
                "where_to_find": "https://www.autopsy.com/download/",
                "how_to_use": "1. Create new case\n2. Add data source\n3. Run ingest modules\n4. Analyze results",
                "features": ["Timeline viewer", "Keyword search", "File recovery", "Web artifact analysis"]
            }
        ]
    },
    "Mobile phone": {
        "tools": [
            {
                "name": "Cellebrite UFED",
                "description": "Industry-standard mobile device forensics platform.",
                "use_cases": ["Data Extraction", "Password Bypass", "App Analysis", "Call Log Recovery"],
                "where_to_find": "https://www.cellebrite.com/en/ufed/",
                "how_to_use": "1. Connect device\n2. Select extraction method\n3. Perform extraction\n4. Analyze extracted data.",
                "features": ["Physical extraction", "Logical extraction", "App data parsing", "Report generation"]
            },
            {
                "name": "Oxygen Forensic Detective",
                "description": "Advanced mobile forensics software.",
                "use_cases": ["Cloud Data Extraction", "Social Media Analysis", "GPS Data Analysis"],
                "where_to_find": "https://www.oxygen-forensic.com/",
                "how_to_use": "1. Connect device\n2. Choose extraction type\n3. Extract data\n4. Analyze using built-in tools.",
                "features": ["Cloud analyzer", "Timeline analysis", "Social media extraction", "Password recovery"]
            }
        ]
    },
    "Smartwatch": {
        "tools": [
            {
                "name": "MSAB Extract",
                "description": "Tool for extracting and analyzing data from wearable devices, including smartwatches.",
                "use_cases": ["Data Recovery", "Activity Logs Analysis", "GPS Data Extraction"],
                "where_to_find": "https://www.msab.com/",
                "how_to_use": "1. Connect the smartwatch to your computer\n2. Use MSAB Extract to pull data\n3. Analyze activity logs and location data.",
                "features": ["Location tracking", "Activity logs", "Health data analysis"]
            },
            {
                "name": "Oxygen Forensic Detective for Wearables",
                "description": "Software for extracting and analyzing data from wearables, including smartwatches.",
                "use_cases": ["Health Data Analysis", "GPS Data Recovery", "Notifications and Call Log Extraction"],
                "where_to_find": "https://www.oxygen-forensic.com/",
                "how_to_use": "1. Connect smartwatch\n2. Select appropriate extraction type\n3. Analyze extracted data for health and location information.",
                "features": ["Cloud data integration", "Health metrics analysis", "Timeline reconstruction"]
            }
        ]
    },
    "Computer Keyboard": {
        "tools": [
            {
                "name": "Keyboard Logger Analysis",
                "description": "Tools for analyzing data from keystroke loggers.",
                "use_cases": ["Keystroke Analysis", "Key Mapping Recovery", "Forensic Investigation of Key Inputs"],
                "where_to_find": "https://www.example-keylogger-tool.com",
                "how_to_use": "1. Extract data from keystroke logger\n2. Use the tool to decode and analyze keystroke patterns\n3. Reconstruct typed information.",
                "features": ["Key mapping visualization", "Timeline analysis", "Reconstruction of key sequences"]
            },
            {
                "name": "Keyboard Activity Monitor",
                "description": "Software for monitoring and analyzing keyboard input patterns.",
                "use_cases": ["Activity Tracking", "Behavioral Analysis", "Detection of Suspicious Key Patterns"],
                "where_to_find": "https://www.keyboard-activity-monitor.com",
                "how_to_use": "1. Install on a forensic workstation\n2. Import keyboard data\n3. Analyze input trends and patterns.",
                "features": ["Activity logs", "Behavioral analysis", "Pattern detection"]
            }
        ]
    },
    "Digital Clock": {
        "tools": [
            {
                "name": "Clock Data Extractor",
                "description": "A tool designed to extract logs and event data from digital clocks, including power cycles, alarms, and time synchronization details.",
                "use_cases": ["Analysis of power-on/off events", "Retrieval of alarm settings and timed events", "Examination of time synchronization logs"],
                "where_to_find": "https://www.clockdataextractor.com",
                "how_to_use": "1. Connect the digital clock to the computer via USB.\n2. Launch the Clock Data Extractor and select the connected device.\n3. Extract event logs and analyze power cycle data.",
                "features": ["Power cycle analysis", "Alarm retrieval", "Time synchronization monitoring"]
            },
            {
                "name": "TimeSync Analyzer",
                "description": "A tool used for analyzing the synchronization of digital clocks with external time sources, ensuring accurate time settings.",
                "use_cases": ["Verification of time sync logs", "Analysis of discrepancies between actual time and clock time", "Correlation of clock events with other time-stamped data"],
                "where_to_find": "https://www.timesyncanalyzer.com",
                "how_to_use": "1. Connect the digital clock to the forensic workstation.\n2. Import time synchronization logs.\n3. Compare with known time servers to detect anomalies.",
                "features": ["Sync accuracy analysis", "Cross-reference with network time sources", "Detailed reporting of time drift"]
            },
            {
                "name": "Digital Clock Event Logger",
                "description": "Software to monitor and log user interactions with digital clocks, such as alarm setting and manual adjustments.",
                "use_cases": ["Tracking user interactions with the clock", "Detecting changes in time settings", "Monitoring for suspicious resets or adjustments"],
                "where_to_find": "https://www.digitalclockeventlogger.com",
                "how_to_use": "1. Connect the clock to a computer running the Digital Clock Event Logger.\n2. Configure the software to monitor changes and generate logs.\n3. Review logs for suspicious time changes.",
                "features": ["Interaction logging", "Real-time monitoring", "Comprehensive event reports"]
            }
        ]
    },
    "Calculator":{
        "tools": [
        {
            "name": "Hex Workshop",
            "description": "A hex editor that helps in analyzing and editing the binary data of digital calculators.",
            "use_cases": ["Data Recovery", "Memory Analysis", "Modification of Calculator Data"],
            "where_to_find": "https://www.hexworkshop.com/",
            "how_to_use": "1. Connect the calculator to your forensic workstation.\n2. Use Hex Workshop to access and analyze memory dumps from the calculator.\n3. Edit binary data and recover deleted information.",
            "features": ["Memory dump analysis", "Binary data editing", "Hexadecimal comparison"]
        },
        {
            "name": "XRY Logical",
            "description": "A forensic analysis tool for extracting data from various digital devices, including calculators.",
            "use_cases": ["Data Extraction", "Analysis of Stored Calculations", "Retrieving Stored History"],
            "where_to_find": "https://www.msab.com/xry-logical/",
            "how_to_use": "1. Connect the calculator to the forensic workstation.\n2. Use XRY Logical to extract all possible data, including stored calculations.\n3. Analyze the data and export results in a readable format.",
            "features": ["Logical extraction", "Detailed reporting", "Supports multiple devices"]
        },
        {
            "name": "DataPilot Forensic",
            "description": "Tool designed to extract and analyze data from calculators and other small electronic devices.",
            "use_cases": ["Data Recovery", "Forensic Analysis of Calculation History", "Timestamp Verification"],
            "where_to_find": "https://www.datapilot.com/forensic/",
            "how_to_use": "1. Connect the calculator via USB to the forensic software.\n2. Use DataPilot to extract stored data, including recent calculations and history.\n3. Verify timestamps of inputs and analyze calculation patterns.",
            "features": ["Timestamp analysis", "Comprehensive data extraction", "Visualization of stored data"]
            }
        ]
    },
    "Camera":{
        "tools": [
        {
            "name": "ExifTool",
            "description": "A powerful tool for reading, writing, and editing metadata in image and video files.",
            "use_cases": ["Metadata Analysis", "GPS Data Extraction", "Image History Verification"],
            "where_to_find": "https://exiftool.org/",
            "how_to_use": "1. Install ExifTool on your forensic workstation.\n2. Use the command line to extract metadata from images.\n3. Analyze data like GPS coordinates, timestamps, and camera model information.",
            "features": ["Metadata extraction", "Batch processing", "Supports various file formats"]
        },
        {
            "name": "Amped FIVE",
            "description": "A forensic video enhancement software for analyzing camera footage.",
            "use_cases": ["Video Enhancement", "Image Clarification", "Frame Analysis"],
            "where_to_find": "https://ampedsoftware.com/five",
            "how_to_use": "1. Import video footage into Amped FIVE.\n2. Apply filters to enhance image clarity.\n3. Analyze frames and generate reports for evidence.",
            "features": ["Frame-by-frame analysis", "Image deblurring", "Export reports"]
        },
        {
            "name": "Magnet AXIOM",
            "description": "Digital forensics software for analyzing data from digital cameras, including image recovery.",
            "use_cases": ["Image Recovery", "Data Carving", "Evidence Correlation"],
            "where_to_find": "https://www.magnetforensics.com/products/axion/",
            "how_to_use": "1. Connect the camera storage to the workstation.\n2. Use AXIOM to scan for recoverable images.\n3. Correlate recovered images with other digital evidence.",
            "features": ["Data recovery", "Artifact analysis", "Detailed reporting"]
            }
        ]
    },
    "Computer Mouse":{
         "tools": [
        {
            "name": "Mouse Activity Logger",
            "description": "A tool for recording and analyzing mouse movements and clicks.",
            "use_cases": ["Activity Monitoring", "User Behavior Analysis", "Click Pattern Recognition"],
            "where_to_find": "https://www.mouseactivitylogger.com/",
            "how_to_use": "1. Install the logger on the forensic workstation.\n2. Connect the computer mouse.\n3. Record and analyze mouse activity to understand user behavior.",
            "features": ["Click tracking", "Movement visualization", "Detailed log reports"]
        },
        {
            "name": "USB Forensic Toolkit",
            "description": "A tool for analyzing data exchanges between USB-connected devices, including computer mice.",
            "use_cases": ["USB Data Analysis", "Device Interaction Tracking", "Mouse Firmware Investigation"],
            "where_to_find": "https://www.usbforensics.com/toolkit",
            "how_to_use": "1. Connect the computer mouse via USB.\n2. Use the toolkit to monitor data exchanges between the mouse and the computer.\n3. Analyze logs for unauthorized interactions or modifications.",
            "features": ["Data capture", "Packet analysis", "Interaction logs"]
        },
        {
            "name": "Mouse Forensics Pro",
            "description": "Advanced tool for extracting and analyzing data from computer mouse devices, including movement history and firmware details.",
            "use_cases": ["Movement History Analysis", "Firmware Investigation", "User Interaction Patterns"],
            "where_to_find": "https://www.mouseforensicspro.com/",
            "how_to_use": "1. Connect the mouse to the forensic workstation.\n2. Extract movement logs and firmware data.\n3. Analyze user patterns and verify firmware integrity.",
            "features": ["Firmware extraction", "Movement pattern analysis", "Integrity checks"]
        }
        ]
    },
    "Headphones":{
        "tools": [
        {
            "name": "Audio Forensics Lab",
            "description": "A tool for analyzing audio data from connected headphones, including audio logs and signal analysis.",
            "use_cases": ["Audio Signal Analysis", "Voice Detection", "Background Noise Analysis"],
            "where_to_find": "https://www.audioforensicslab.com/",
            "how_to_use": "1. Connect the headphones to your forensic workstation.\n2. Use the tool to record audio and analyze the signal.\n3. Extract voice patterns and analyze background noise for further investigation.",
            "features": ["Signal visualization", "Frequency analysis", "Noise reduction"]
        },
        {
            "name": "USB Audio Analyzer",
            "description": "A tool for monitoring and analyzing USB data from headphones, focusing on audio quality and data integrity.",
            "use_cases": ["Data Integrity Analysis", "Audio Quality Testing", "USB Data Monitoring"],
            "where_to_find": "https://www.usbaudioanalyzer.com/",
            "how_to_use": "1. Connect USB headphones to the forensic workstation.\n2. Use the analyzer to monitor data transfer and audio quality.\n3. Analyze for any data loss or manipulation in the audio stream.",
            "features": ["Real-time monitoring", "Bitrate analysis", "Data integrity checks"]
        },
        {
            "name": "Bluetooth Packet Sniffer",
            "description": "A tool for capturing and analyzing Bluetooth communication between headphones and paired devices.",
            "use_cases": ["Bluetooth Signal Analysis", "Device Pairing Investigation", "Security Protocol Verification"],
            "where_to_find": "https://www.bluetoothpacketsniffer.com/",
            "how_to_use": "1. Pair the Bluetooth headphones with the forensic workstation.\n2. Capture communication packets using the sniffer.\n3. Analyze pairing details and potential security flaws.",
            "features": ["Packet capture", "Encryption analysis", "Bluetooth version compatibility"]
        }
        ]
    },
    "ipod":{
         "tools": [
        {
            "name": "Audio Forensics Lab",
            "description": "A tool for analyzing audio data from connected headphones, including audio logs and signal analysis.",
            "use_cases": ["Audio Signal Analysis", "Voice Detection", "Background Noise Analysis"],
            "where_to_find": "https://www.audioforensicslab.com/",
            "how_to_use": "1. Connect the headphones to your forensic workstation.\n2. Use the tool to record audio and analyze the signal.\n3. Extract voice patterns and analyze background noise for further investigation.",
            "features": ["Signal visualization", "Frequency analysis", "Noise reduction"]
        },
        {
            "name": "USB Audio Analyzer",
            "description": "A tool for monitoring and analyzing USB data from headphones, focusing on audio quality and data integrity.",
            "use_cases": ["Data Integrity Analysis", "Audio Quality Testing", "USB Data Monitoring"],
            "where_to_find": "https://www.usbaudioanalyzer.com/",
            "how_to_use": "1. Connect USB headphones to the forensic workstation.\n2. Use the analyzer to monitor data transfer and audio quality.\n3. Analyze for any data loss or manipulation in the audio stream.",
            "features": ["Real-time monitoring", "Bitrate analysis", "Data integrity checks"]
        },
        {
            "name": "Bluetooth Packet Sniffer",
            "description": "A tool for capturing and analyzing Bluetooth communication between headphones and paired devices.",
            "use_cases": ["Bluetooth Signal Analysis", "Device Pairing Investigation", "Security Protocol Verification"],
            "where_to_find": "https://www.bluetoothpacketsniffer.com/",
            "how_to_use": "1. Pair the Bluetooth headphones with the forensic workstation.\n2. Capture communication packets using the sniffer.\n3. Analyze pairing details and potential security flaws.",
            "features": ["Packet capture", "Encryption analysis", "Bluetooth version compatibility"]
        }
        ]
    },
    "Microphone":{
    "tools": [
            {
                "name": "Audio Forensics Lab",
                "description": "A tool for analyzing audio data from connected headphones, including audio logs and signal analysis.",
                "use_cases": ["Audio Signal Analysis", "Voice Detection", "Background Noise Analysis"],
                "where_to_find": "https://www.audioforensicslab.com/",
                "how_to_use": "1. Connect the headphones to your forensic workstation.\n2. Use the tool to record audio and analyze the signal.\n3. Extract voice patterns and analyze background noise for further investigation.",
                "features": ["Signal visualization", "Frequency analysis", "Noise reduction"]
            },
            {
                "name": "USB Audio Analyzer",
                "description": "A tool for monitoring and analyzing USB data from headphones, focusing on audio quality and data integrity.",
                "use_cases": ["Data Integrity Analysis", "Audio Quality Testing", "USB Data Monitoring"],
                "where_to_find": "https://www.usbaudioanalyzer.com/",
                "how_to_use": "1. Connect USB headphones to the forensic workstation.\n2. Use the analyzer to monitor data transfer and audio quality.\n3. Analyze for any data loss or manipulation in the audio stream.",
                "features": ["Real-time monitoring", "Bitrate analysis", "Data integrity checks"]
            },
            {
                "name": "Bluetooth Packet Sniffer",
                "description": "A tool for capturing and analyzing Bluetooth communication between headphones and paired devices.",
                "use_cases": ["Bluetooth Signal Analysis", "Device Pairing Investigation", "Security Protocol Verification"],
                "where_to_find": "https://www.bluetoothpacketsniffer.com/",
                "how_to_use": "1. Pair the Bluetooth headphones with the forensic workstation.\n2. Capture communication packets using the sniffer.\n3. Analyze pairing details and potential security flaws.",
                "features": ["Packet capture", "Encryption analysis", "Bluetooth version compatibility"]
            }
        ]
    },
    "Musical Keyboard":{
        "tools": [
        {
            "name": "MIDI Analyzer",
            "description": "Tool for capturing and analyzing MIDI data from musical keyboards, useful for understanding played sequences.",
            "use_cases": ["MIDI Data Capture", "Sequence Analysis", "Note Detection"],
            "where_to_find": "https://www.midianalyzer.com/",
            "how_to_use": "1. Connect the musical keyboard via MIDI to the forensic workstation.\n2. Use the analyzer to capture and inspect note sequences.\n3. Identify played notes, timing, and sequence structure.",
            "features": ["Note visualization", "Tempo analysis", "Sequence mapping"]
        },
        {
            "name": "Audio Spectrogram Viewer",
            "description": "Software that visualizes and analyzes sound frequencies from musical keyboards, revealing the audio signature of each note.",
            "use_cases": ["Sound Signature Analysis", "Frequency Mapping", "Harmonic Content Detection"],
            "where_to_find": "https://www.audiospectrogramviewer.com/",
            "how_to_use": "1. Connect the keyboard to your forensic workstation.\n2. Play sequences and capture sound frequencies.\n3. Analyze the harmonic content and frequency patterns.",
            "features": ["Frequency spectrum display", "Harmonic analysis", "Audio fingerprinting"]
        },
        {
            "name": "Keylogger for Musical Instruments",
            "description": "Specialized tool for recording and analyzing keystrokes on digital musical keyboards.",
            "use_cases": ["Key Stroke Recording", "Note Frequency Analysis", "Timing Verification"],
            "where_to_find": "https://www.keyloggerforinstruments.com/",
            "how_to_use": "1. Connect the digital keyboard to the forensic workstation.\n2. Monitor keystrokes to capture and analyze musical notes.\n3. Verify note timing and detect anomalies.",
            "features": ["Keystroke logging", "Timing visualization", "Note frequency analysis"]
        }
        ]
    },
    "Television":{
         "tools": [
        {
            "name": "HDMI Data Analyzer",
            "description": "A tool for capturing and analyzing HDMI signals from smart TVs, helpful in identifying display content.",
            "use_cases": ["Content Analysis", "Signal Quality Testing", "Input Source Identification"],
            "where_to_find": "https://www.hdmidataanalyzer.com/",
            "how_to_use": "1. Connect the analyzer to the television's HDMI output.\n2. Capture and examine signal data.\n3. Identify displayed content and input sources.",
            "features": ["HDMI signal capture", "Quality testing", "Input tracking"]
        },
        {
            "name": "TV Log Inspector",
            "description": "A software tool that accesses and analyzes activity logs from smart TVs, providing insights into app usage and streaming history.",
            "use_cases": ["Usage Analysis", "App Interaction Tracking", "Streaming History Review"],
            "where_to_find": "https://www.tvloginspector.com/",
            "how_to_use": "1. Connect to the smart TV’s storage or cloud account.\n2. Extract logs of app interactions and streaming history.\n3. Review for any unauthorized usage or content playback.",
            "features": ["Log extraction", "App usage tracking", "Streaming history"]
        },
        {
            "name": "Screen Mirroring Monitor",
            "description": "Tool for monitoring and capturing screen mirroring sessions from connected devices to a television.",
            "use_cases": ["Screen Mirroring Detection", "Content Analysis", "Connected Device Verification"],
            "where_to_find": "https://www.screenmirroringmonitor.com/",
            "how_to_use": "1. Enable screen mirroring and capture the content displayed on the television.\n2. Use the tool to track the connected devices and monitor mirrored sessions.\n3. Analyze the content for investigative purposes.",
            "features": ["Mirroring session capture", "Device tracking", "Content monitoring"]
        }
        ]
    },
    "Tripod":{
         "tools": [
        {
            "name": "Image Stabilization Analyzer",
            "description": "Tool for analyzing the stability of images or videos captured with the help of tripods, detecting shake levels.",
            "use_cases": ["Image Stability Assessment", "Shake Detection", "Tripod Usage Verification"],
            "where_to_find": "https://www.imagestabilizationanalyzer.com/",
            "how_to_use": "1. Import the footage or images captured with the tripod.\n2. Use the tool to analyze stability and detect any movement.\n3. Verify tripod usage based on shake levels.",
            "features": ["Stability metrics", "Shake visualization", "Movement tracking"]
        },
        {
            "name": "Tripod Position Tracker",
            "description": "Tool for analyzing tripod placement and angle data, often used in forensic photography and videography setups.",
            "use_cases": ["Position Analysis", "Angle Verification", "Scene Recreation"],
            "where_to_find": "https://www.tripodpositiontracker.com/",
            "how_to_use": "1. Use the tool to capture tripod position data from forensic scenes.\n2. Analyze angles and positioning for accurate scene recreation.\n3. Generate reports on positioning consistency.",
            "features": ["Angle detection", "Position mapping", "Scene validation reports"]
        },
        {
            "name": "Footprint and Mark Analysis Tool",
            "description": "A tool for analyzing marks or footprints left by tripods, useful for scene correlation and alignment verification.",
            "use_cases": ["Mark Detection", "Scene Correlation", "Alignment Verification"],
            "where_to_find": "https://www.footprintmarkanalyzer.com/",
            "how_to_use": "1. Capture images of marks left by the tripod on surfaces.\n2. Use the tool to correlate marks with positioning data.\n3. Validate the alignment and setup based on footprint patterns.",
            "features": ["Mark visualization", "Footprint comparison", "Alignment analysis"]
        }
        ]
    },
    "Vehicle registration plate":{
        "tools": [
        {
            "name": "Automatic License Plate Recognition (ALPR)",
            "description": "Software for detecting and recognizing license plates in images or videos, ideal for forensic applications.",
            "use_cases": ["License Plate Detection", "Character Recognition", "Vehicle Tracking"],
            "where_to_find": "https://www.alpr.com/",
            "how_to_use": "1. Import images or videos containing license plates.\n2. Use the tool’s ALPR capabilities to detect and recognize characters.\n3. Identify vehicle registration numbers for further investigation.",
            "features": ["High accuracy OCR", "Batch processing", "Vehicle tracking support"]
        },
        {
            "name": "Plate Region Enhancer",
            "description": "Tool for enhancing the visibility and clarity of vehicle registration plates in low-resolution or poor-quality images.",
            "use_cases": ["Image Enhancement", "Contrast Adjustment", "Noise Reduction"],
            "where_to_find": "https://www.plateenhancer.com/",
            "how_to_use": "1. Import images containing vehicle plates.\n2. Apply enhancement filters to increase clarity.\n3. Use the improved image for accurate character recognition.",
            "features": ["Noise reduction", "Contrast enhancement", "Sharpness adjustment"]
        },
        {
            "name": "Vehicle Database Lookup",
            "description": "A database tool that cross-references license plate numbers with registration information for owner identification.",
            "use_cases": ["Owner Identification", "Registration Verification", "Vehicle History Tracking"],
            "where_to_find": "https://www.vehicledatabase.com/",
            "how_to_use": "1. Enter the recognized license plate number.\n2. Retrieve the vehicle’s registration information and ownership details.\n3. Cross-check with other case information for verification.",
            "features": ["Instant lookup", "Owner details retrieval", "Historical tracking"]
        }
        ]
    }
}



# Helper function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to save user data to a JSON file
def save_user_data(username, password, name, is_admin=False, blocked=False):
    hashed_password = hash_password(password)
    users = load_user_data()
    users["usernames"][username] = {
        "name": name,
        "password": hashed_password,
        "is_admin": is_admin,
        "blocked": blocked
    }
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

# Load user data from JSON file
def load_user_data():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {"usernames": {}}

# Load the YOLOv8 model
@st.cache_resource
def load_yolov8_model():
    model = YOLO("C:/Users/nhari/OneDrive/Desktop/new/yolov8x-oiv7.pt")  # Update this path to your model location
    return model

model = load_yolov8_model()

# Function for making predictions
def make_prediction_yolo(img):
    results = model(img)
    return results

# Function to save user history after detection
def save_user_history(username, img_path, results):
    user_history = get_user_history(username, include_deleted=True)
    annotated_img = results[0].plot()
    annotated_img_path = img_path.replace(".png", "_annotated.png")
    Image.fromarray(annotated_img).save(annotated_img_path)

    boxes = results[0].boxes
    readable_results = []
    if boxes is not None:
        data = boxes.data.cpu().numpy()
        for box in data:
            label = model.names[int(box[5])]
            confidence = box[4]
            readable_results.append({
                "label": label,
                "confidence": round(float(confidence), 2)
            })
    
    user_history["uploads"].append({
        "img_path": img_path,
        "annotated_img_path": annotated_img_path,
        "results": readable_results,
        "timestamp": datetime.now().isoformat(),
        "deleted": False,
        "deleted_by_admin": False
    })

    user_data_path = f"user_data/{username}.json"
    with open(user_data_path, "w") as f:
        json.dump(user_history, f, indent=4)

# Function to block or unblock a user
def toggle_user_block_status(username):
    users = load_user_data()
    current_status = users["usernames"][username].get("blocked", False)
    users["usernames"][username]["blocked"] = not current_status
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)
    return not current_status

# Function to load user history with an optional filter for deleted records
def get_user_history(username, include_deleted=False):
    path = f"user_data/{username}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            history = json.load(f)
            if not include_deleted:
                history["uploads"] = [
                    upload for upload in history["uploads"]
                    if not upload.get("deleted", False) and not upload.get("deleted_by_admin", False)
                ]
            return history
    return {"uploads": []}

# Function to mark a specific user upload as deleted by admin
def delete_user_upload_by_admin(username, upload_indices):
    user_history = get_user_history(username, include_deleted=True)
    for idx in upload_indices:
        user_history["uploads"][idx]["deleted_by_admin"] = True
    with open(f"user_data/{username}.json", "w") as f:
        json.dump(user_history, f, indent=4)
    st.success(f"Selected uploads have been deleted for {username}.")


# Function to delete a specific upload for a user
def delete_user_upload(username, upload_index):
    user_data_path = f"user_data/{username}.json"
    
    if os.path.exists(user_data_path):
        with open(user_data_path, "r") as f:
            user_history = json.load(f)
        
        # Check if the upload index is valid
        if 0 <= upload_index < len(user_history["uploads"]):
            user_history["uploads"][upload_index]["deleted"] = True  # Mark the upload as deleted

            # Save the updated history back to the file
            with open(user_data_path, "w") as f:
                json.dump(user_history, f, indent=4)
            
            st.success(f"Upload {upload_index + 1} has been deleted for {username}.")
        else:
            st.warning(f"Invalid upload index: {upload_index}")
    else:
        st.warning(f"No upload history found for {username}.")


# Function to delete all uploads for a specific user
def delete_all_user_uploads(username):
    user_data_path = f"user_data/{username}.json"
    
    if os.path.exists(user_data_path):
        with open(user_data_path, "r") as f:
            user_history = json.load(f)
        
        # Mark all uploads as deleted
        for upload in user_history.get("uploads", []):
            upload["deleted"] = True
        
        # Save the updated history back to the file
        with open(user_data_path, "w") as f:
            json.dump(user_history, f, indent=4)
        
        st.success(f"All uploads have been deleted for {username}.")
    else:
        st.warning(f"No upload history found for {username}.")



# Ensure the dedicated reports directory exists
reports_dir = "reports"
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)


# Define the StyledReportPDF class with PDF styling
class StyledReportPDF(FPDF):
    def header(self):
        # Slide-like colorful header
        self.set_fill_color(50, 75, 160)  # Dark blue
        self.set_text_color(255, 255, 255)  # White
        self.set_font("Arial", "B", 20)
        self.cell(0, 15, "Object Detection Report", ln=True, align="C", fill=True)
        self.ln(10)

    def add_slide_title(self, title):
        # Main title in larger, bold font
        self.set_text_color(50, 75, 160)  # Dark blue
        self.set_font("Arial", "B", 18)
        self.cell(0, 10, title, ln=True, align="L")
        self.ln(5)

    def add_section_header(self, header):
        # Section headers with background color
        self.set_fill_color(220, 220, 255)  # Light blue
        self.set_text_color(0, 0, 128)  # Dark blue
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, header, ln=True, fill=True)
        self.ln(5)
        self.set_text_color(0, 0, 0)

    def add_bullet_point(self, text, bullet="*"):  # Replacing • with *
        # Bullet points with simple ASCII characters
        self.set_font("Arial", "", 12)
        self.cell(10, 10, bullet, align="R")
        self.cell(0, 10, text, ln=True)



    def add_text(self, text, bold=False, italic=False):
        # Regular text, with optional bold and italic
        style = ""
        if bold:
            style += "B"
        if italic:
            style += "I"
        self.set_font("Arial", style, 12)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def add_forensic_tool_section(self, tools):
        # Forensic tools styled with background and bullets
        for idx, tool in enumerate(tools, start=1):
            self.set_fill_color(240, 240, 255)  # Soft light background
            self.cell(0, 10, f"{idx}. {tool['name']}", ln=True, fill=True)
            self.add_text(tool["description"], bold=True)
            self.add_text("Use Cases:")
            for use_case in tool["use_cases"]:
                self.add_bullet_point(use_case)
            self.add_text("How to Use:", bold=True)
            self.add_text(tool["how_to_use"])
            self.add_text("Features:", bold=True)
            for feature in tool["features"]:
                self.add_bullet_point(feature)
            self.ln(5)

    def add_image_section(self, img_path, annotated_img_path):
        # Add the original and annotated image side by side
        self.add_section_header("Original and Annotated Images")
        self.image(img_path, x=10, y=self.get_y(), w=90)
        self.image(annotated_img_path, x=110, y=self.get_y(), w=90)
        self.ln(95)

    def add_detection_summary(self, results):
        # Detection summary with styled list of objects
        self.add_section_header("Detection Summary")
        if results:
            for idx, result in enumerate(results, start=1):
                label = result.get("label", "Unknown")
                confidence = result.get("confidence", "N/A")
                self.add_text(f"{idx}. {label} (Confidence: {confidence}%)", bold=True)
        else:
            self.add_text("No objects detected.")

    def add_non_digital_evidence_section(self, non_digital_evidence):
        # Non-digital evidence with descriptions styled in a soft background
        self.add_section_header("Non-Digital Evidence")
        for label, description in non_digital_evidence.items():
            self.set_fill_color(245, 245, 255)  # Slightly colored background
            self.cell(0, 10, label, ln=True, fill=True)
            self.add_text(description, italic=True)


# Example usage for debugging
img_path = os.path.join("uploads_dir", "original.png")
annotated_img_path = os.path.join("uploads_dir", "annotated.png")

# Check if both images exist before proceeding
if os.path.exists(img_path) and os.path.exists(annotated_img_path):
    report_path = generate_report(
        username="test_user",
        img_path=img_path,
        annotated_img=annotated_img_path,
        results=[{"label": "Laptop", "confidence": 0.95}]
    )
    print(f"Report generated at: {report_path}")
else:
    print("One or both image files not found. Ensure images are saved at the specified paths.")



# Mapping evidence types to their specific forensic tool functions
evidence_tool_functions = {
    "laptop": laptop_tools,
    "calculator": calculator_tools,
    "camera": camera_tools,
    "computer keyboard": computer_keyboard_tools,
    "computer mouse": computer_mouse_tools,
    "digital clock": digital_clock_tools,
    "headphones": headphones_tools,
    "ipod": ipod_tools,
    "microphone": microphone_tools,
    "mobile phone": mobile_phone_tools,
    "musical keyboard": musical_keyboard_tools,
    "smartwatch": smartwatch_tools,
    "television": television_tools,
    "tripod": tripod_tools,
    "vehicle registration plate": vehicle_registration_plate_tools,
    "watch": watch_tools
}

# Function to generate a report and save it in the reports directory
@st.cache
def generate_report(username, img_path, annotated_img, results):
    # Define the report path within the reports directory
    report_path = os.path.join(reports_dir, f"{username}_detection_report_{int(datetime.now().timestamp())}.pdf")
    pdf = StyledReportPDF()  # Use StyledReportPDF for consistent styling
    pdf.add_page()

    # Title Section
    pdf.add_slide_title("Detection Report")

    # Image Section: Save the annotated image if not already saved
    pdf.add_section_header("Uploaded Image")
    annotated_img_path = img_path.replace(".png", "_annotated.png")
    annotated_img.save(annotated_img_path)  # Save the annotated image
    pdf.add_image_section(img_path, annotated_img_path)

    # Detection Details Section
    pdf.add_section_header("Detection Details")
    if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
        data = results[0].boxes.data.cpu().numpy()
        digital_evidence = []
        non_digital_evidence = []

        for box in data:
            label = model.names[int(box[5])].capitalize()
            confidence = round(box[4] * 100, 2)
            entry = f"{label} (Confidence: {confidence}%)"
            pdf.add_text(entry)

            # Categorize the evidence type for later sections
            if label.lower() in evidence_tool_functions:
                digital_evidence.append((label, confidence))
            else:
                non_digital_evidence.append((label, confidence))
    else:
        pdf.add_text("No objects detected.")

    # Digital Evidence Tools Section
    pdf.add_section_header("Digital Evidence and Forensic Tools")
    if digital_evidence:
        for label, confidence in digital_evidence:
            pdf.add_text(f"{label} (Confidence: {confidence}%)")

            # Call the relevant tool function if available
            tool_function = evidence_tool_functions.get(label.lower())
            if tool_function:
                try:
                    forensic_tools = tool_function()
                    pdf.add_forensic_tool_section(forensic_tools)  # Display all tools for each evidence
                except Exception as e:
                    pdf.add_text(f"Error retrieving tools for {label}: {e}")
            else:
                pdf.add_text("No tools available for this evidence type.")
    else:
        pdf.add_text("No digital evidence detected.")

    # Non-Digital Evidence Section
    if non_digital_evidence:
        pdf.add_section_header("Non-Digital Evidence")
        for label, confidence in non_digital_evidence:
            pdf.add_text(f"{label} (Confidence: {confidence}%)")
            description = NON_DIGITAL_EVIDENCE_DESC.get(label, "No specific forensic details available.")
            pdf.add_text(description)

    # Save the report in the reports directory
    pdf.output(report_path)
    return report_path




# Authentication system
st.sidebar.title("User Authentication")
users = load_user_data()

# Session state initialization
def init_session_state():

    if 'register_mode' not in st.session_state:
        st.session_state.register_mode = False
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None
    if 'show_tools' not in st.session_state:
        st.session_state.show_tools = False
    if 'selected_evidence' not in st.session_state:
        st.session_state.selected_evidence = None
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'report_path' not in st.session_state:
        st.session_state.report_path = None
    if 'report_viewed' not in st.session_state:
        st.session_state.report_viewed = False
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    if 'detected' not in st.session_state:
        st.session_state.detected = False
    if 'img_path' not in st.session_state:
        st.session_state.img_path = None

init_session_state()

# Registration logic
if st.session_state.register_mode:
    st.sidebar.subheader("Register")
    register_username = st.sidebar.text_input("Username", key="register_username")
    register_name = st.sidebar.text_input("Full Name", key="register_name")
    register_password = st.sidebar.text_input("Password", type="password", key="register_password")

    # Check current number of admins
    current_admin_count = len([user for user in users["usernames"].values() if user.get("is_admin", False)])
    is_admin = st.sidebar.checkbox("Register as Admin")

    if st.sidebar.button("Submit Registration"):
        # Perform validation checks
        if not register_username or not register_name or not register_password:
            st.sidebar.error("All fields are required for registration.")
        elif register_username in users["usernames"]:
            st.sidebar.error("Username already exists. Please choose a different one.")
        elif is_admin and current_admin_count >= 2:
            st.sidebar.error("Only a maximum of 2 admins are allowed. The limit is reached for admin accounts.")
        else:
            save_user_data(register_username, register_password, register_name, is_admin)
            st.sidebar.success("Registration successful! You can now log in.")
            st.session_state.register_mode = False  # Return to login mode

# Show login form if not in registration mode and not logged in
if not st.session_state.register_mode and not st.session_state.logged_in:
    st.sidebar.subheader("Login")
    login_username = st.sidebar.text_input("Username", key="login_username")
    login_password = st.sidebar.text_input("Password", type="password", key="login_password")

    if st.sidebar.button("Login"):
        if login_username in users["usernames"] and \
           users["usernames"][login_username]["password"] == hash_password(login_password):
            if users["usernames"][login_username].get("blocked", False):
                st.sidebar.error("Your account is blocked. Please contact the administrator.")
            else:
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.session_state.is_admin = users["usernames"][login_username].get("is_admin", False)
                st.sidebar.success(f"Welcome, {users['usernames'][login_username]['name']}!")
        else:
            st.sidebar.error("Invalid username or password.")

# Toggle registration mode on button click
if not st.session_state.logged_in:
    if st.sidebar.button("Switch to Register" if not st.session_state.register_mode else "Back to Login"):
        st.session_state.register_mode = not st.session_state.register_mode

# Display logged-in status
if st.session_state.logged_in:
    st.sidebar.success(f"Welcome, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.is_admin = False
        st.sidebar.info("You have been logged out.")


# Admin Interface: Manage Users and their Data
if st.session_state.logged_in and st.session_state.is_admin:
    st.header("User Management")
    
    for username in users["usernames"]:
        user_details = users["usernames"][username]
        user_history = get_user_history(username, include_deleted=True)
        num_uploads = len(user_history["uploads"])

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"Username: {username} - Name: {user_details['name']} - Uploads: {num_uploads}")

        block_status = "Unblock" if user_details.get("blocked", False) else "Block"
        with col2:
            if st.button(f"{block_status} User", key=f"block_{username}"):
                new_status = toggle_user_block_status(username)
                st.success(f"{'Blocked' if new_status else 'Unblocked'} user {username}.")

        with col3:
            if st.button("View Details", key=f"view_{username}"):
                st.session_state.selected_user = username

    # Display detailed upload history for the selected user
    if st.session_state.selected_user:
        selected_user = st.session_state.selected_user
        user_history = get_user_history(selected_user, include_deleted=True)
        st.subheader(f"Details for {selected_user}")

        selected_indices = st.multiselect("Select uploads to delete", options=range(len(user_history["uploads"])), format_func=lambda x: f"Upload {x+1}")
        if selected_indices and st.button("Delete Selected Uploads"):
            delete_user_upload_by_admin(selected_user, selected_indices)
            st.session_state.selected_user = selected_user  # Refresh the view

        if user_history["uploads"]:
            for idx, upload in enumerate(user_history["uploads"]):
                st.markdown("---")
                if upload.get("deleted_by_admin", False):
                    st.write("⚠ This upload was deleted by the admin.")
                elif upload.get("deleted", False):
                    st.write("🗑 This upload was deleted by the user.")
                else:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.image(upload["img_path"], caption="Input Image", width=300)
                    with col2:
                        st.image(upload["annotated_img_path"], caption="Output Image", width=300)
                    with col3:
                        if st.button("Delete", key=f"delete_{selected_user}_{idx}"):
                            delete_user_upload_by_admin(selected_user, [idx])
                            st.session_state.selected_user = selected_user  # Refresh the view
                    
                    st.write("### Detected Objects:")
                    display_detection_results(upload["results"], is_history=True)

                    # View Report section for admin
                    report_path = upload.get("report_path")
                    if report_path and os.path.exists(report_path):
                        with st.expander("View Report"):
                            with open(report_path, "rb") as report_file:
                                st.download_button(
                                    label="Download Report",
                                    data=report_file,
                                    file_name=f"{selected_user}_upload_{idx + 1}_report.pdf",
                                    mime="application/pdf"
                                )
                            st.write("You can view or download the report above.")
                    else:
                        st.write("Report not available for this upload.")
        else:
            st.write("No uploads found for this user.")

            


# Detection and history section for regular users
if st.session_state.logged_in and not st.session_state.is_admin:
    st.markdown("## Evidence Detection")
    user_data_path = f"user_data/{st.session_state.username}.json"
    uploads_dir = "uploads"
    if not os.path.exists("user_data"):
        os.makedirs("user_data")
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Initialize session states if they don't exist
    if 'show_tools' not in st.session_state:
        st.session_state.show_tools = False
    if 'selected_evidence' not in st.session_state:
        st.session_state.selected_evidence = None
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    if 'detected' not in st.session_state:
        st.session_state.detected = False
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'report_viewed' not in st.session_state:
        st.session_state.report_viewed = False
    if 'report_path' not in st.session_state:
        st.session_state.report_path = None
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    if 'img_path' not in st.session_state:
        st.session_state.img_path = None
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None

    # Define uploads directory
    uploads_dir = "uploads"
    if not os.path.exists("user_data"):
        os.makedirs("user_data")
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Evidence Detection Section
    st.markdown("## Evidence Detection")

    # Upload Section
    col1, col2 = st.columns([3, 1])
    with col1:
        upload = st.file_uploader("Upload Image Here:", type=["png", "jpg", "jpeg"])


    # When image is uploaded
    if upload:
        img = Image.open(upload).convert("RGB")
        st.session_state.uploaded_image = img
        st.session_state.uploaded = True
        st.session_state.detected = False
        st.session_state.report_generated = False
        st.session_state.report_viewed = False
        st.session_state.report_path = None
        
        # Save the image with a unique name based on timestamp
        img_path = f"{uploads_dir}/{st.session_state.username}_{datetime.now().timestamp()}.png"
        img.save(img_path)
        st.session_state.img_path = img_path

        # Check if image file was saved successfully
        if not os.path.exists(img_path):
            print("Image file could not be saved at:", img_path)
        else:
            print("Image saved successfully at:", img_path)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # Detection Section
    if st.session_state.uploaded and not st.session_state.detected:
        if st.button("Detect Evidence"):
            results = make_prediction_yolo(st.session_state.uploaded_image)
            st.session_state.detection_results = results
            save_user_history(st.session_state.username, st.session_state.img_path, results)
            st.session_state.detected = True

            # Annotate and save the image with bounding boxes
            annotated_img = results[0].plot()
            st.session_state.img_with_bboxes = Image.fromarray(annotated_img)
            
            # Save annotated image for future use
            annotated_img_path = st.session_state.img_path.replace(".png", "_annotated.png")
            st.session_state.img_with_bboxes.save(annotated_img_path)
            
            # Check if annotated image was saved
            if not os.path.exists(annotated_img_path):
                print("Annotated image could not be saved at:", annotated_img_path)
            else:
                print("Annotated image saved successfully at:", annotated_img_path)


    # Display detection results and report options if detection has been done
    if st.session_state.detected:
        st.header("📊 Detection Results")
        results = st.session_state.detection_results
        annotated_img = results[0].plot()
        img_with_bboxes = Image.fromarray(annotated_img)
        st.image(img_with_bboxes, caption="Detected Objects", use_column_width=True)

        # Report Generation Section
        if st.session_state.detected:
            with st.expander("Generate & Manage Report"):
                # Generate Report Button - Only visible if report not yet generated
                if not st.session_state.report_generated:
                    if st.button("Generate Report"):
                        if os.path.exists(st.session_state.img_with_bboxes):
                            report_path = generate_report(
                                st.session_state.username,
                                st.session_state.img_path,
                                st.session_state.img_with_bboxes,  # Use saved bboxes image in session state
                                st.session_state.detection_results
                            )
                            st.session_state.report_path = report_path
                            st.session_state.report_generated = True
                            st.success("Your PDF report has been generated successfully!")
                        else:
                            print("Error: Annotated image file not found at:", annotated_img_path)

                # Download and View Report Buttons - Only visible after report generation
                if st.session_state.report_generated:
                    col1, col2 = st.columns(2)
                    with col1:
                        if os.path.exists(st.session_state.report_path):
                            with open(st.session_state.report_path, "rb") as f:
                                st.download_button("Download Report", f, file_name="detection_report.pdf", key="download_report_btn")
                        else:
                            print("Error: Report file not found at:", st.session_state.report_path)
                    
                    with col2:
                        if st.button("View Report", key="view_report_btn"):
                            st.session_state.report_viewed = True

            # Report Viewing Section
            if st.session_state.report_viewed:
                if os.path.exists(st.session_state.report_path):
                    with fitz.open(st.session_state.report_path) as pdf_doc:
                        num_pages = pdf_doc.page_count
                        for page_num in range(num_pages):
                            page = pdf_doc[page_num]
                            image = page.get_pixmap()
                            st.image(image.tobytes(), caption=f"Page {page_num + 1}")
                else:
                    print("Error: Report file not found at:", st.session_state.report_path)

            # Display detection results details
            display_detection_results(results, is_history=False, unique_id="live_detection")
        
   # History Section
    col1, col2 = st.columns([2, 6])
    with col1:
        if st.button("View My History"):
            st.session_state.show_history = not st.session_state.show_history
            st.session_state.show_tools = False
            st.session_state.selected_evidence = None

    with col2:
        if st.button("Hide History"):
            st.session_state.show_history = False
            st.session_state.show_tools = False
            st.session_state.selected_evidence = None

    if st.session_state.show_history:
        st.header("Your Upload History")
        user_history = get_user_history(st.session_state.username, include_deleted=True)

        # Add "Delete All" button
        if user_history["uploads"]:
            if st.button("Delete All Uploads"):
                delete_all_user_uploads(st.session_state.username)
                st.session_state.show_history = True
                st.session_state.show_tools = False
                st.session_state.selected_evidence = None

        # Display individual uploads in reverse order (latest at the top)
        if user_history["uploads"]:
            for idx, upload in enumerate(reversed(user_history["uploads"])):
                st.markdown("---")
                if upload.get("deleted_by_admin", False):
                    st.write("This upload was deleted by the admin.")
                elif upload.get("deleted", False):
                    st.write("This upload was deleted by you.")
                else:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.image(upload["img_path"], caption="Input Image", width=300)
                    with col2:
                        st.image(upload["annotated_img_path"], caption="Output Image", width=300)
                    with col3:
                        if st.button("Delete", key=f"user_delete_{idx}"):
                            delete_user_upload(st.session_state.username, len(user_history["uploads"]) - idx - 1)
                            st.session_state.show_history = True
                            st.session_state.show_tools = False
                            st.session_state.selected_evidence = None

                    # Expander for generating and downloading report for each history item
                    with st.expander(f"Generate & Manage Report for Detection {idx+1}"):
                        if 'report_path_history' not in st.session_state:
                            st.session_state.report_path_history = {}

                        # Generate report only if not previously generated for this history item
                        if st.session_state.report_path_history.get(len(user_history["uploads"]) - idx - 1) is None:
                            report_path = generate_report(
                                st.session_state.username,
                                upload["img_path"],
                                Image.open(upload["annotated_img_path"]),
                                upload["results"]
                            )
                            st.session_state.report_path_history[len(user_history["uploads"]) - idx - 1] = report_path

                        # Display download button for the generated report
                        with open(st.session_state.report_path_history[len(user_history["uploads"]) - idx - 1], "rb") as f:
                            st.download_button(f"Download Report for Detection {idx+1}", f, file_name=f"detection_report_{idx+1}.pdf")

                    st.write("### Detected Objects:")
                    display_detection_results(upload["results"], is_history=True, unique_id=f"history_{len(user_history['uploads']) - idx - 1}")
        else:
            st.write("No uploads found.")




            
# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.write("Evidence Detector")
    st.sidebar.write("© PES UNIVERSITY")
