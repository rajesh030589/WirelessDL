#!/bin/bash

# Run the TX1 Transmitter script in the remote computer

# Message generation
echo 'Transmitter generates message and transmits'
sshpass -p "July2021!" ssh rajesh@10.145.83.81 ". ~/.profile; cd /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/; python3 TX_Feedback_Encoder1.py"

# Message transfer to receiver
sshpass -p "July2021!" scp rajesh@10.145.83.81:/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/TX.bin /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/
echo 'Message transferred to receiver'

# Receiver receives first message
python3 RX_Feedback_Decoder1.py
echo 'Messages received at the receiver'

# Transmitter process kill
sshpass -p "July2021!" ssh rajesh@10.145.83.81 'python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/flowgraph_process_kill.py'

# first Feedback prepared by receiver
python3 RX_Feedback_Encoder1.py
echo 'Receiver sent first feedback to transmitter'

sshpass -p "July2021!" ssh rajesh@10.145.83.81 ". ~/.profile; cd /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/; python3 TX_Feedback_Decoder1.py"
echo 'Transmitter received first feedback'

# Receiver transmitter process killed
python3 flowgraph_process_kill.py

# Feedback received and second transmission prepared
echo 'Transmitter prepares second transmission'
sshpass -p "July2021!" ssh rajesh@10.145.83.81 ". ~/.profile; cd /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/; python3 TX_Feedback_Encoder2.py"


# Receiver receives second message
python3 RX_Feedback_Decoder2.py
echo 'Messages received at the receiver for the second time'

# Transmitter process kill
sshpass -p "July2021!" ssh rajesh@10.145.83.81 'python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/flowgraph_process_kill.py'


# final feedback Feedback prepared by receiver
python3 RX_Feedback_Encoder2.py
echo 'Receiver sent final feedback to transmitter'

sshpass -p "July2021!" ssh rajesh@10.145.83.81 ". ~/.profile; cd /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/; python3 TX_Feedback_Decoder2.py"
echo 'Transmitter received final feedback'

# Receiver transmitter process killed
python3 flowgraph_process_kill.py

# Feedback received and second transmission prepared
echo 'Transmitter prepares final transmission'
sshpass -p "July2021!" ssh rajesh@10.145.83.81 ". ~/.profile; cd /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/; python3 TX_Feedback_Encoder3.py"


# Receiver receives final message
python3 RX_Feedback_Decoder3.py
echo 'Messages received at the receiver for the second time'

# Transmitter process kill
sshpass -p "July2021!" ssh rajesh@10.145.83.81 'python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/flowgraph_process_kill.py'

# Receiver does final decoding
python3 RX_Feedback_Encoder3.py
echo 'Messages Decoded'

