#!/usr/bin/env Rscript

#the purpose of this code is to use the warbleR package to extract acoustic features from vocalizations clips
#this script requires that all wav clips be in a single directory

# set directories
#wavs.dir is the location of all of the wav clips to be analyzed
#analysis.dir is the directory to save the csv containing the features for each wav clip
argv<-commandArgs(trailingOnly=TRUE)
species<-argv[1]
wavs.dir<-argv[2]
analysis.dir<-argv[3]

print(species)
print(wavs.dir)
print(analysis.dir)

#load required packages
x <- gc()
x <- c( "parallel", "bioacoustics", "warbleR","pbapply")
aa <- lapply(x, function(y) {
        if(!y %in% installed.packages()[,"Package"])  {
                install.packages(y)
        }
        try(require(y, character.only = T), silent = T)
})

#set warbleR params
#these parameters have been chosen for Peromyscus pups
print("setting warbleR options.")
warbleR_options(wav.path = wavs.dir, wl = 1024, flim = c(5, 125), ovlp = 25, bp = c(5, 125))

#save the wav clips in an est file (aka selection table) - this is the file format that warbleR uses to reference the wav clips
print("making selection table.")
est <- selection_table(whole.recs = T, extended = F, confirm.extended = F, pb = F)

#get acoustic features
#print("calculating acoustic features.")
#sp <- specan(est,harmonicity = FALSE, fast = TRUE, pb = F)

#get sound pressure level
print("getting sound pressure levels.")
pressure <- sound_pressure_level(est, pb = F)

#merge
print("merging data.")
#data <- merge(sp, pressure)
data <- pressure

print("making dataframe.")
prms <- data.frame(est[, c("sound.files")], data)
colnames(prms)[colnames(prms) == 'sound.files'] <- 'source_file'
prms <- subset(prms, select=-c(sound.files.1, selec))

print("writing csv.")
write.csv(prms,file.path(analysis.dir,paste(species,"warbler_features.csv",sep="")),row.names = FALSE)
print("done.")
