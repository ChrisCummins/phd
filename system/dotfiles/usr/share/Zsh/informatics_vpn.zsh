# Connecting to the Edinburgh Informatics VPN.
#
# Setup
# =====
#
#   1. $ ~/.mkdir -pv ~/.local/share/openvpn
#   2. Download Informatics-InfNets-Forum.ovpn from:
#      http://computing.help.inf.ed.ac.uk/openvpn-config-files
#   3. $ mv ~/Downloads/Informatics-InfNets-Forum.ovpn ~/.local/share/openvpn/
#   4. Create the file ~/.local/share/openvpn/Informatics-credentials.txt with:
#        <username>
#        <password>
#   5. $ chmod 600 ~/.local/share/openvpn/*
#
# Usage
# =====
#
# Connect to Informatics VPN using openvpn:
#
#    $ inf_vpn_connect
#
# Check if the VPN is running:
#
#    $ inf_vpn_status
#
# Disconnect from the Informatics VPN:
#
#    $ inf_vpn_disconnect

inf_vpn_connect() {
	sudo openvpn \
		--config ~/.local/share/openvpn/Informatics-InfNets-Forum.ovpn \
		--auth-user-pass ~/.local/share/openvpn/Informatics-credentials.txt \
		--daemon
}

inf_vpn_status() {
	if [[ "$(ps auxww | grep openvpn | wc -l)" == 2 ]]; then
		echo "running"
	else
		echo "not running"
		return 1
	fi
}

inf_vpn_disconnect() {
	sudo killall -SIGINT openvpn
}
