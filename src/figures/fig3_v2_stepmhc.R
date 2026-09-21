RESULTS <- "results/bench_common/"      # SET FIRST - HANDOFF_9 section 8
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(tidyr);library(patchwork)})

LAB <- c(stepmhc_paired="STEP-MHC (paired)", stepmhc_beta="STEP-MHC (\u03b2-only)",
         nettcr="NetTCR-2.2", unipmt="UniPMT", pmtnet="pMTnet",
         teim="TEIM", panpep="PanPep", imrex="ImRex")
ORD <- names(LAB)
th <- theme_bw(base_size=8) + theme(
  panel.grid.minor=element_blank(),
  plot.title=element_text(face="bold",size=10,hjust=0),
  plot.margin=margin(5,7,5,5))

b <- read.csv(paste0(RESULTS,"metrics_breakdown.csv")) %>% mutate(model=factor(model,ORD,LAB))

hm <- function(lv,ttl,xlab){
  d <- b %>% filter(level==lv) %>% group_by(key) %>% mutate(nmax=max(pos)) %>% ungroup() %>%
       mutate(key2=sprintf("%s\n(n=%d)",key,nmax))
  d$key2 <- factor(d$key2, d %>% distinct(key2,nmax) %>% arrange(desc(nmax)) %>% pull(key2))
  ggplot(d,aes(key2,model,fill=roc)) +
    geom_tile(colour="white",linewidth=.6) +
    geom_text(aes(label=sprintf("%.2f",roc),colour=abs(roc-.5)>.22),size=2.1,show.legend=FALSE) +
    scale_colour_manual(values=c("grey15","white")) +
    scale_fill_gradient2(low="#3B6FB6",mid="grey95",high="#C2185B",midpoint=.5,
                         limits=c(0,1),name="ROC-AUC") +
    scale_y_discrete(limits=rev(unname(LAB))) +
    labs(title=ttl,x=xlab,y=NULL) +
    th + theme(axis.text.x=element_text(angle=45,hjust=1,size=6),
               axis.text.y=element_text(size=6.5),
               legend.key.width=unit(7,"pt"),legend.key.height=unit(20,"pt"),
               legend.title=element_text(size=6.5),legend.text=element_text(size=6),
               panel.grid=element_blank())
}
pa <- hm("allele","A","HLA allele (n = positives)")
pb <- hm("patient","B","Patient (n = positives)")

ctl <- data.frame(model=factor(c("nettcr","unipmt","pmtnet","teim","panpep","imrex"),ORD),
                  own=c(0.9607,0.9658,0.8375,0.7428,0.5271,0.7337),
                  bench=c(0.5076,0.4907,0.4899,0.5094,0.5121,0.4857)) %>%
  pivot_longer(c(own,bench),names_to="set",values_to="roc") %>%
  mutate(set=factor(set,c("own","bench"),c("own data (positive control)","BNT221 neoantigens")))
pc <- ggplot(ctl,aes(model,roc,fill=set)) +
  geom_col(position=position_dodge(.78),width=.72) +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey40",linewidth=.3) +
  geom_text(aes(label=sprintf("%.2f",roc)),position=position_dodge(.78),vjust=-.55,size=2.0) +
  scale_fill_manual(values=c("own data (positive control)"="#7E8AA2",
                             "BNT221 neoantigens"="#D9534F"),name=NULL) +
  scale_x_discrete(labels=LAB) + coord_cartesian(ylim=c(0,1.15)) +
  labs(title="C",x=NULL,y="ROC-AUC") +
  th + theme(panel.border=element_rect(colour="grey20",fill=NA,linewidth=.4),
             panel.grid.major.x=element_blank(),
             legend.position="inside",legend.position.inside=c(.62,.90),
             legend.direction="horizontal",legend.key.size=unit(7,"pt"),
             legend.text=element_text(size=6),
             legend.background=element_rect(fill=alpha("white",.85),colour=NA),
             axis.text.x=element_text(angle=45,hjust=1,size=6.5))

reg <- data.frame(
  model=rep(c("NetTCR-2.2","STEP-MHC"),each=2),
  task=factor(rep(c("Viral, peptide-rich\n(NetTCR-2.2 split)","Neoantigen\n(BNT221)"),2),
              c("Viral, peptide-rich\n(NetTCR-2.2 split)","Neoantigen\n(BNT221)")),
  roc=c(0.961,0.508,0.815,0.670))
pd <- ggplot(reg,aes(task,roc,fill=model)) +
  geom_col(position=position_dodge(.7),width=.6) +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey40",linewidth=.3) +
  geom_text(aes(label=sprintf("%.3f",roc)),position=position_dodge(.7),vjust=-.55,size=2.1) +
  scale_fill_manual(values=c("NetTCR-2.2"="#B5A8D5","STEP-MHC"="#C2185B"),name=NULL) +
  coord_cartesian(ylim=c(0,1.15)) + labs(title="D",x=NULL,y="ROC-AUC") +
  th + theme(panel.border=element_rect(colour="grey20",fill=NA,linewidth=.4),
             panel.grid.major.x=element_blank(),
             legend.position="inside",legend.position.inside=c(.5,.92),
             legend.direction="horizontal",legend.key.size=unit(7,"pt"),
             legend.text=element_text(size=6.5),legend.background=element_blank(),
             axis.text.x=element_text(size=6.5))

neg <- data.frame(
  scheme=factor(c("Random\nMHC + TCR\n(their protocol)","Peptide-swap\nreal MHC + TCR\n(this study)"),
                c("Random\nMHC + TCR\n(their protocol)","Peptide-swap\nreal MHC + TCR\n(this study)")),
  roc=c(0.9658,0.5932), pr=c(0.7613,0.2270))
pe <- ggplot(neg,aes(scheme,roc,fill=scheme)) +
  geom_col(width=.55) +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey40",linewidth=.3) +
  geom_text(aes(label=sprintf("%.3f",roc)),vjust=-.55,size=2.3) +
  annotate("segment",x=1,xend=2,y=1.12,yend=1.12,linewidth=.3,colour="grey30") +
  annotate("text",x=1.5,y=1.19,label="\u0394 = 0.37 AUC",size=2.4,fontface="bold") +
  scale_fill_manual(values=c("#7E8AA2","#D9534F")) +
  coord_cartesian(ylim=c(0,1.28)) +
  labs(title="E",x=NULL,y="ROC-AUC") +
  th + theme(panel.border=element_rect(colour="grey20",fill=NA,linewidth=.4),
             panel.grid.major.x=element_blank(), legend.position="none",
             axis.text.x=element_text(size=6), axis.title.y=element_text(size=7))

fig <- pa/pb/(pc)/(pd|pe) + plot_layout(heights=c(1,0.95,0.95,0.95))
ggsave("figures/naming_v2/Fig3_breakdowns_stepmhc.pdf",fig,width=180,height=230,units="mm",device=cairo_pdf)
ggsave("figures/naming_v2/Fig3_breakdowns_stepmhc.png",fig,width=180,height=230,units="mm",dpi=600)
cat("wrote figures/naming_v2/Fig3_breakdowns_stepmhc\n")
