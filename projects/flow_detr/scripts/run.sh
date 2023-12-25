chmod +777 /root/detrex/scripts/run_script.sh
# model_name, backbone, epoch, tasks, tasks_pattern, (model_checkpoint)
exec /root/detrex/scripts/run_script.sh detr flowformer 50ep train install
#exec /root/detrex/scripts/run_script.sh flow_detr flowformer 50ep analyze param model_final.pth
#exec /root/detrex/scripts/run_script.sh flow_detr r50_fine_turning 50ep train
#exec /root/detrex/scripts/run_script.sh flow_detr r50_fine_turning 300ep analyze flop model_0004999.pth
#exec /root/detrex/scripts/run_script.sh flow_detr r50_fine_turning 300ep analyze structure
