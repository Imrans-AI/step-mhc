# FIGURE — clinical validation and deployment metrics
# Panels: a per-cohort percentiles | b permutation null | c forest + numeric columns
#         d neoantigen-selection enrichment | e TCR-selection enrichment
#         f NeoTCR external validation
# Style follows src/fig_pooled.R (DAISY Adv Sci conventions). Vector PDF, 600 dpi raster.
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(patchwork); library(tidyr)
})
dir.create("figures", showWarnings = FALSE)

COL  <- c(BNT221="#B5A8D5", HBV="#A3C9E8", Caushi="#EBA8C3", NSCLC="#9FD5C8")
EDGE <- c(BNT221="#8377A8", HBV="#6D95B5", Caushi="#B87694", NSCLC="#6FA79A")
ORDER <- c("BNT221","HBV","Caushi","NSCLC")
ACC  <- "#7E6BA8"; ACC2 <- "#4E7FA8"

theme_pub <- function(base = 7) {
  theme_bw(base_size = base, base_family = "Helvetica") +
    theme(panel.grid = element_blank(),
          panel.border = element_rect(colour="black", linewidth=0.35),
          axis.ticks = element_line(colour="black", linewidth=0.35),
          axis.text = element_text(colour="black", size=base),
          axis.title = element_text(colour="black", size=base+1),
          plot.title = element_text(size=base+1, hjust=0),
          plot.tag = element_text(size=base+4, face="bold"),
          legend.position = "none",
          plot.margin = margin(4,6,4,4))
}

O <- read.csv("results/pooled/observed_percentiles.csv", stringsAsFactors=FALSE)
O$cohort <- factor(O$cohort, levels=ORDER)
ND_ALL <- read.csv("results/pooled/null_draws.csv")$null   # REAL permutation draws
NULLV  <- median(ND_ALL)
set.seed(1)
boot_ci <- function(x,B=5000){s<-replicate(B,median(sample(x,length(x),replace=TRUE)))
  c(lo=unname(quantile(s,.025)),hi=unname(quantile(s,.975)))}

S <- O %>% group_by(cohort) %>%
  summarise(n=n(), ant=n_distinct(antigen), med=median(pct),
            lo=boot_ci(pct)[1], hi=boot_ci(pct)[2], .groups="drop")
allci <- boot_ci(O$pct)
S <- bind_rows(S, tibble(cohort="Pooled", n=nrow(O), ant=n_distinct(O$antigen),
                         med=median(O$pct), lo=allci[1], hi=allci[2]))
S$cohort <- factor(S$cohort, levels=c("Pooled", ORDER))

LABF <- setNames(sprintf("%s\nn=%d", c("BNT221","HBV+ HCC","Caushi","TIL NSCLC"),
                         S$n[match(ORDER, S$cohort)]), ORDER)

# a — per-cohort distributions
pa <- ggplot(O, aes(1, pct, fill=cohort, colour=cohort)) +
  geom_hline(yintercept=NULLV, linetype="dashed", linewidth=0.32, colour="grey35") +
  geom_boxplot(width=0.55, outlier.shape=NA, linewidth=0.32, colour="grey40", alpha=0.8) +
  geom_jitter(width=0.16, size=0.62, alpha=0.6, stroke=0) +
  facet_grid(~cohort, labeller=labeller(cohort=LABF)) +
  scale_fill_manual(values=COL) + scale_colour_manual(values=EDGE) +
  scale_y_continuous(breaks=seq(0,1,.25), expand=c(0.02,0)) +
  scale_x_continuous(limits=c(0.4,1.6)) +
  coord_cartesian(ylim=c(0,1)) +   # clip the VIEW, never drop data (was silently dropping pct=1.0)
  labs(x=NULL, y="Percentile vs\nnon-reactive TIL") + theme_pub() +
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(),
        strip.background=element_blank(), strip.text=element_text(size=6.2, colour="grey20"),
        panel.spacing=unit(3,"pt"))

# b — permutation null
ND <- ND_ALL; obs <- median(O$pct)   # real draws, not simulated
pb <- ggplot(data.frame(x=ND), aes(x)) +
  geom_histogram(bins=32, fill="grey82", colour="grey55", linewidth=0.25) +
  geom_vline(xintercept=obs, colour=ACC, linewidth=0.8) +
  annotate("text", x=obs, y=Inf, label=sprintf("observed %.3f", obs),
           size=2.0, hjust=1.08, vjust=1.6, colour=ACC, angle=90) +
  annotate("text", x=NULLV, y=Inf, label=sprintf("null %.3f", NULLV), size=2.0, hjust=0.5,
           vjust=1.5, colour="grey45") +
  annotate("text", x=-Inf, y=Inf, label="P < 0.002", size=2.3, hjust=-0.25, vjust=1.8,
           fontface="bold", colour="grey20") +
  labs(x="Median percentile under permutation", y="Shuffles") + theme_pub()

# c — forest
pc <- ggplot(S, aes(med, cohort, colour=cohort, fill=cohort)) +
  geom_vline(xintercept=NULLV, linetype="dashed", linewidth=0.32, colour="grey35") +
  geom_errorbar(aes(xmin=lo, xmax=hi), orientation="y", width=0, linewidth=1.1) +
  geom_point(aes(shape=cohort=="Pooled"), size=2.4, stroke=0.3, colour="grey25") +
  scale_shape_manual(values=c(`TRUE`=23,`FALSE`=21)) +
  scale_fill_manual(values=c(COL, Pooled="grey30")) +
  scale_colour_manual(values=c(EDGE, Pooled="grey30")) +
  scale_y_discrete(limits=rev(levels(S$cohort)),
    labels=c("BNT221","HBV+ HCC","Caushi","TIL NSCLC","Pooled")[
      match(rev(levels(S$cohort)), c(ORDER,"Pooled"))]) +
  scale_x_continuous(limits=c(0,1), breaks=seq(0,1,.2), expand=c(0.02,0)) +
  labs(x="Median percentile (95% CI)", y=NULL) + theme_pub()

# d/e — enrichment ladders
mk_enrich <- function(f, title, xlab, col) {
  E <- read.csv(f)
  ggplot(E, aes(factor(k), rate)) +
    geom_col(fill=col, colour=colorspace::darken(col,0.25), linewidth=0.3, width=0.62) +
    geom_errorbar(aes(ymin=lo, ymax=hi), width=0.16, linewidth=0.32, colour="grey25") +
    geom_segment(aes(x=as.numeric(factor(k))-0.31, xend=as.numeric(factor(k))+0.31,
                     y=chance, yend=chance), linetype="dashed", linewidth=0.4,
                 colour="grey25") +
    geom_text(aes(y=hi, label=sprintf("%.1f\u00d7", enrich)), vjust=-0.7, size=2.1,
              fontface="bold", colour="grey20") +
    scale_y_continuous(limits=c(0, max(E$hi)*1.25), expand=c(0,0)) +
    labs(x=xlab, y="Validated targets captured", title=title) +
    annotate("text", x=Inf, y=Inf, label="dashed = chance", size=1.9, hjust=1.08,
             vjust=1.8, colour="grey40") +
    theme_pub()
}
pd_ <- mk_enrich("results/bnt221/route3_enrichment.csv",
                 "Neoantigen selection (32 candidates)", "top k", COL["BNT221"])
pe  <- mk_enrich("results/pooled/route4_enrichment.csv",
                 "TCR selection (601 candidates)", "top k", COL["HBV"])

# f — NeoTCR external
N <- read.csv("results/neotcr/neotcr_percentiles.csv")
pf <- ggplot(N, aes(1, pct)) +
  geom_hline(yintercept=0.552, linetype="dashed", linewidth=0.32, colour="grey35") +
  geom_boxplot(width=0.4, outlier.shape=NA, linewidth=0.32, colour="grey40",
               fill="#C9B8DE", alpha=0.85) +
  geom_jitter(width=0.12, size=0.8, alpha=0.65, stroke=0, colour=EDGE["BNT221"]) +
  annotate("text", x=1.38, y=0.552, label="null 0.552", size=2.0, hjust=0,
           vjust=-0.6, colour="grey45") +
  annotate("text", x=1.38, y=median(N$pct), label=sprintf("%.3f\nP = 0.002", median(N$pct)),
           size=2.1, hjust=0, fontface="bold", colour="grey20", lineheight=0.95) +
  scale_y_continuous(limits=c(0,1), breaks=seq(0,1,.25), expand=c(0.02,0)) +
  scale_x_continuous(limits=c(0.6,2.3)) +
  labs(x=NULL, y="Percentile",
       title=sprintf("NeoTCR external\nn=%d, %d epitopes", nrow(N),
                     length(unique(N$peptide)))) +
  theme_pub() + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())

fig <- (pa) / (pc | pb) / (pd_ | pe | pf) +
  plot_layout(heights=c(0.62, 1.0, 1.0)) +
  plot_annotation(tag_levels="A",
    caption="Negatives matched on peptide, HLA and repertoire origin. Dashed line = permutation null, never 0.5.",
    theme=theme(plot.caption=element_text(size=6.2, hjust=0, colour="grey25")))

ggsave("figures/naming_v2/Fig4_clinical_upper.pdf", fig, width=183, height=185, units="mm", device=cairo_pdf)
ggsave("figures/naming_v2/Fig4_clinical_upper.png", fig, width=183, height=185, units="mm", dpi=600)
cat("wrote figures/naming_v2/Fig4_clinical_upper.{pdf,png}\n")
