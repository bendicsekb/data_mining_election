library(ggplot2)
library(mice)
library(tidyverse)

# Working directory
setwd("C:/Users/20200059/Documents/Github/data_mining_election/Synthetic/")

data <- read.csv("reversed.csv")
data <- read.csv("pairwise.csv")
data <- read.csv("lasttofirst.csv")
names(data)

pd <- position_dodge(.65)
plot <- data %>%
  rename("nrows" = names(data)[1]) %>%
  gather("measure", "time", -c(nrows, ndescs, ntargets)) %>%
  mutate(nrows = as.factor(nrows)) %>%
  mutate(ntargets = as.factor(ntargets)) %>%
  mutate(ndescs = as.factor(ndescs)) %>%
  mutate(measure = ordered(measure,
                           levels = c("none", "entropy", "sqrtnN", "norm", "labelwise", "pairwise"))) %>%
  #filter(ntargets == 32) %>%
  filter(ndescs == 32) %>%
  #filter(measure %in% c('entropy','norm')) %>%
  ggplot(aes(x=measure, y=time, color=nrows, shape=ntargets)) +
  geom_point(position = pd) + 
  scale_colour_manual(values = c("#a6cee3", "#1f78b4", "#b2df8a")) + 
  scale_shape_manual(values=c(4, 15, 2)) +
  ylab('Runtime in seconds') + 
  xlab('Quality measure') +
  #labs(title = '') + 
  theme_bw() + 
  theme(plot.title = element_text(vjust=-4), 
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        #panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(size=4),
        axis.title = element_text(size=10),
        legend.title = element_text(size = 6), 
        legend.text  = element_text(size = 4),
        #legend.position="bottom",
        #legend.justification="right",
        #legend.margin=margin(-20, 0, 0, 0))
  )
plot
name <- paste('timereversed.pdf', sep = "", collapse = NULL)
name <- paste('timepairwise.pdf', sep = "", collapse = NULL)
name <- paste('timelasttofirst.pdf', sep = "", collapse = NULL)
ggsave(name, width = 20, height = 7, units = "cm")
ggsave(name, width = 6, height = 8, units = "cm")