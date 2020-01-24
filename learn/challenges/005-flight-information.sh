# Given a string matrix containing the information of running flights
# (columns like source, destination, arrival time, departure time
# ...), use any tools known to you to ..  Extract the info of all
# flights from source X to destination Y.
#
# Extract a summary info of how many flights leave from each source
# and how many flights arrive at each destination.  Justify your
# choice for the specific tool / language you would use.

echo "Departures:"
awk '{print $1;}' <flight-information-test.txt | uniq -c | sort -rn
echo
echo "Arrivals:"
awk '{print $2;}' <flight-information-test.txt | uniq -c | sort -rn

# Justification:
#
# Use common UNIX tools. Easy and quick to develop and modify. There's
# no need to over-engineer. If performance becomes an issue, it's easy
# to profile.
#
# Can be pipeline-parallelised using GNU parallel:
#
#   echo
#   echo "Arrivals:"
#   awk '{print $2;}' < flight-information-test.txt \
#       | parallel --pipe uniq -c \
#       | parallel --pipe sort -rn
