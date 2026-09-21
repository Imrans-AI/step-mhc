RESULTS <- "results/bench_common/"      # SET FIRST - HANDOFF_9 section 8
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(patchwork)})

LAB <- c(stepmhc_paired="STEP-MHC (paired)", stepmhc_beta="STEP-MHC (\u03b2-only)",
         nettcr="NetTCR-2.2", unipmt="UniPMT", pmtnet="pMTnet",
         teim="TEIM", panpep="PanPep", imrex="ImRex")
ORD <- names(LAB)
PAL <- c(stepmhc_paired="#C2185B", stepmhc_beta="#E57398", nettcr="#B5A8D5",
         unipmt="#8FB8DE", pmtnet="#A3C9E8", teim="#EBA8C3",
         panpep="#9FD5C8", imrex="#C9C9C9")
th <- theme_bw(base_size=8) + theme(
  panel.grid.minor=element_blank(), panel.grid.major.x=element_blank(),
  legend.position="none", plot.title=element_text(face="bold",size=9,hjust=0),
  plot.margin=margin(5,7,5,5))

m  <- read.csv(paste0(RESULTS,"metrics_main.csv"))      %>% mutate(model=factor(model,ORD))
b  <- read.csv(paste0(RESULTS,"metrics_breakdown.csv")) %>% mutate(model=factor(model,ORD))
cv <- read.csv(paste0(RESULTS,"curve_points.csv"))      %>% mutate(model=factor(model,ORD))

bar <- function(df,yv,ttl,ylab,lo,hi,nl)
  ggplot(df, aes(model,.data[[yv]],fill=model)) + geom_col(width=.7) +
    geom_hline(yintercept=nl,linetype="dashed",colour="grey40",linewidth=.3) +
    geom_text(aes(label=sprintf("%.3f",.data[[yv]])),vjust=-.55,size=2.0) +
    scale_fill_manual(values=PAL) + scale_x_discrete(labels=LAB) +
    coord_cartesian(ylim=c(0,hi)) + labs(title=ttl,x=NULL,y=ylab) +
    th + theme(axis.text.x=element_text(angle=45,hjust=1,size=6.2))
pa <- bar(m,"roc","A","ROC-AUC",0,0.82,0.5)
pb <- bar(m,"pr", "B","PR-AUC",0,0.42,114/684)

# curves: legend BELOW the panel, never over the data
mk <- function(k){z <- cv %>% distinct(model,curve,auc) %>% filter(curve==k) %>%
                       arrange(match(model,ORD))
                  setNames(sprintf("%s (%.3f)",LAB[as.character(z$model)],z$auc),z$model)}
curve <- function(k,ttl,xl,yl,hl)
  ggplot(filter(cv,curve==k),aes(x,y,colour=model)) +
    {if(k=="roc") geom_abline(slope=1,intercept=0,linetype="dashed",colour="grey60",linewidth=.3)
     else geom_hline(yintercept=hl,linetype="dashed",colour="grey60",linewidth=.3)} +
    geom_line(linewidth=.5) +
    scale_colour_manual(values=PAL,labels=mk(k),name=NULL) +
    guides(colour=guide_legend(nrow=3,byrow=TRUE)) +
    coord_cartesian(ylim=c(0,1)) + labs(title=ttl,x=xl,y=yl) +
    th + theme(legend.position="bottom", legend.key.size=unit(7,"pt"),
               legend.text=element_text(size=5.6), legend.margin=margin(t=-2),
               legend.box.spacing=unit(2,"pt"))
pc <- curve("roc","C","False positive rate","True positive rate",NA)
pd <- curve("pr","D","Recall","Precision",114/684)

grid <- function(lv,ttl,NR=2){
  d <- b %>% filter(level==lv) %>% group_by(key) %>% mutate(nmax=max(pos)) %>% ungroup() %>%
       mutate(key2=sprintf("%s (n=%d)",key,nmax))
  d$key2 <- factor(d$key2, d %>% distinct(key2,nmax) %>% arrange(desc(nmax)) %>% pull(key2))
  ggplot(d,aes(model,roc,fill=model)) + geom_col(width=.75) +
    geom_hline(yintercept=.5,linetype="dashed",colour="grey40",linewidth=.25) +
    facet_wrap(~key2,nrow=NR) + scale_fill_manual(values=PAL) + scale_x_discrete(labels=LAB) +
    coord_cartesian(ylim=c(0,1)) + labs(title=ttl,x=NULL,y="ROC-AUC") +
    th + theme(axis.text.x=element_text(angle=90,hjust=1,vjust=.5,size=4.6),
               strip.text=element_text(size=5.8,margin=margin(1.5,0,1.5,0)))
}
pe <- grid("allele", "E")
pf <- grid("patient","F",NR=2)

ctl <- data.frame(model=factor(c("nettcr","unipmt","pmtnet","teim","panpep","imrex"),ORD),
                  own=c(0.9607,0.9658,0.8375,0.7428,0.5271,0.7337),
                  bench=c(0.5076,0.4907,0.4899,0.5094,0.5121,0.4857)) %>%
  pivot_longer(c(own,bench),names_to="set",values_to="roc") %>%
  mutate(set=factor(set,c("own","bench"),c("own data (positive control)","BNT221 neoantigens")))
pg <- ggplot(ctl,aes(model,roc,fill=set)) +
  geom_col(position=position_dodge(.78),width=.72) +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey40",linewidth=.3) +
  geom_text(aes(label=sprintf("%.2f",roc)),position=position_dodge(.78),vjust=-.55,size=1.95) +
  scale_fill_manual(values=c("own data (positive control)"="#7E8AA2",
                             "BNT221 neoantigens"="#D9534F"),name=NULL) +
  scale_x_discrete(labels=LAB) + coord_cartesian(ylim=c(0,1.15)) +
  labs(title="G",
       x=NULL,y="ROC-AUC") +
  th + theme(panel.border=element_rect(colour="grey20",fill=NA,linewidth=.4),
             legend.position="inside", legend.position.inside=c(.62,.86),
             legend.direction="horizontal", legend.key.size=unit(7,"pt"),
             legend.text=element_text(size=6),
             legend.background=element_rect(fill=alpha("white",.85),colour=NA),
             axis.text.x=element_text(angle=45,hjust=1,size=6.2))

fig <- (pa|pb)/(pc|pd) + plot_layout(heights=c(1,1.35)) &
       theme(plot.title=element_text(face="bold",size=10,hjust=0),
             plot.margin=margin(5,7,5,5))
ggsave("figures/naming_v2/Fig2_benchmark_stepmhc.pdf",fig,width=180,height=150,units="mm",device=cairo_pdf)
ggsave("figures/naming_v2/Fig2_benchmark_stepmhc.png",fig,width=180,height=150,units="mm",dpi=600)
cat("wrote figures/naming_v2/Fig2_benchmark_stepmhc\n")
