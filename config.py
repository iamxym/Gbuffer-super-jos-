from sched import scheduler


hs=4
ws=4
#previous_frames=4
learningrate=1e-3
batch_size=8
printPeriod=1000
total_epochs=100
scheduler_step=1
scheduler_gamma=0.98
p="D:/TrainingSetCompressed/"
basePaths=["F:/MD_train/"]

# testPaths=["G:/RF_Gbuffer/RF_test/"]
# basePaths=["F:/SS_train_set/MD_train/","F:/SS_train_set/BK_train/","F:/SS_train_set/RF_train/"]
testPaths=["G:/MD_Gbuffer/MD_test/"]

#basePaths=["C:/train_set/RF_TAA_train/"]
#testPaths=["G:/RF_Gbuffer/RF_TAA_test/"]