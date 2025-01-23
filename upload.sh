cd /data
mkdir -p mimic_arena_gz
tar -cf - mimic_arena | pv | pigz | split -b 4GB - mimic_arena_gz/backup.tar.gz.part
cd /home/root/dreamerv3-expertdata-gen
python push_to_hub.py

# apt-get install -y pv pigz