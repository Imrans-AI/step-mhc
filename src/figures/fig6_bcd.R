RESULTS <- "results/"                   # SET FIRST - HANDOFF_9 section 8
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(patchwork);library(ggrepel)})
th <- theme_bw(base_size=8) + theme(panel.grid.minor=element_blank(),
        plot.title=element_text(face="bold",size=10,hjust=0), plot.margin=margin(5,7,5,5))

# B — HERMES native favorability across 113 crystals (NO cross-task bar comparison)
h <- read.csv(paste0(RESULTS,"hermes_triplets.csv"))
pb <- ggplot(h, aes(mean_favorability)) +
  geom_histogram(bins=24, fill="#8377A8", colour="white", linewidth=.25) +
  geom_vline(xintercept=mean(h$mean_favorability), colour="#C2185B", linewidth=.5) +
  annotate("text", x=Inf, y=Inf, hjust=1.05, vjust=1.6, size=2.3,
           label=sprintf("n = %d crystals\nmean %.2f \u00b1 %.2f\nnative-vs-decoy site AUC 0.936",
                         nrow(h), mean(h$mean_favorability), sd(h$mean_favorability))) +
  labs(title="B", x="Mean per-residue native favorability (HERMES)", y="Crystals") + th

# C — depth law
dep <- read.csv(paste0(RESULTS,"immrep/immrep_with_depth.csv"))
ct  <- suppressWarnings(cor.test(dep$train_pos, dep$auroc, method="spearman"))
pc <- ggplot(dep, aes(train_pos, auroc)) +
  geom_hline(yintercept=.5, linetype="dashed", colour="grey40", linewidth=.3) +
  geom_smooth(method="lm", se=TRUE, colour="#C2185B", fill="#C2185B", alpha=.12,
              linewidth=.5, formula=y~x) +
  geom_point(size=1.7, colour="grey20") +
  geom_text_repel(aes(label=peptide), size=1.9, max.overlaps=20, segment.size=.2) +
  scale_x_log10() +
  annotate("text", x=Inf, y=-Inf, hjust=1.05, vjust=-0.8, size=2.4, fontface="bold",
           label=sprintf("Spearman \u03c1 = %.3f, p = %.4f (n = %d)",
                         unname(ct$estimate), ct$p.value, nrow(dep))) +
  labs(title="C", x="Positives per epitope in TRAINING (log scale)",
       y="External ROC-AUC (IMMREP)") + th

# D — lever ceiling
lv <- read.csv(paste0(RESULTS,"six_lever_clean.csv"))
NICE <- c(seq_cnn_ref="CNN baseline", esm_contrastive="ESM-2 contrastive",
          esm_crossattn="ESM-2 cross-attention*", esm_vat="ESM-2 + VAT*",
          lantern="LANTERN", esm_multitask="ESM-2 multitask*",
          contrastive="Two-phase pretrained*")
lv$nice <- NICE[lv$lever]
lv <- rbind(lv[,c("nice","auc","std")], data.frame(nice="STEP-MHC (canonical)", auc=0.581, std=NA))
lv$nice <- factor(lv$nice, lv$nice[order(lv$auc)])
pd_ <- ggplot(lv, aes(nice, auc, fill=nice=="STEP-MHC (canonical)")) +
  geom_col(width=.7) +
  geom_errorbar(aes(ymin=auc-std, ymax=auc+std), width=.2, linewidth=.3, na.rm=TRUE) +
  geom_hline(yintercept=.5, linetype="dashed", colour="grey40", linewidth=.3) +
  geom_text(aes(label=sprintf("%.3f",auc)), hjust=-0.3, size=2.1) +
  scale_fill_manual(values=c(`FALSE`="#B8B8C4", `TRUE`="#C2185B")) +
  coord_flip(ylim=c(0.45,0.64)) +
  labs(title="D", x=NULL, y="Unseen-epitope macro-AUROC") +
  th + theme(legend.position="none", panel.grid.major.y=element_blank(),
             axis.text.y=element_text(size=6.8))

ggsave("figures/Fig6_BCD.pdf", (pb|pc)/pd_ + plot_layout(heights=c(1,1)),
       width=180, height=150, units="mm", device=cairo_pdf)
cat("wrote figures/Fig6_BCD.pdf\n")
