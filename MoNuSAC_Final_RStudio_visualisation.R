suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(xml2)
  library(EBImage)
  library(png)
})

set.seed(42)

DATA_DIR <- file.path(
  Sys.getenv("USERPROFILE"),
  "Downloads",
  "MoNuSAC Testing Data and Annotations",
  "MoNuSAC Testing Data and Annotations"
)
OUT_DIR <- "monusac_output"
PREP_DIR <- file.path(OUT_DIR, "preprocessed")
MASK_DIR <- file.path(PREP_DIR, "masks")
PANEL_DIR <- file.path(OUT_DIR, "image_panels")
CROP_DIR <- file.path(PREP_DIR, "crops_256")
CROP_SIZE <- 256L
MAX_IMAGES <- Inf

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(PREP_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(MASK_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(PANEL_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(CROP_DIR, showWarnings = FALSE, recursive = TRUE)

classes <- c("epithelial", "lymphocyte", "neutrophil", "macrophage")
class_id <- setNames(seq_along(classes), classes)
class_cols <- c(
  epithelial = "#E63946",
  lymphocyte = "#2A9D8F",
  neutrophil = "#F4A261",
  macrophage = "#457B9D"
)
theme_set(theme_minimal(base_size = 12))

save_plot <- function(p, filename, width = 9, height = 5) {
  ggsave(file.path(OUT_DIR, filename), p, width = width, height = height, dpi = 300)
  print(p)
}

warn_skip <- function(...) warning(sprintf(...), call. = FALSE)

# Build a dataset-wide image index so XML files can match images in other subfolders.
image_index <- list.files(
  DATA_DIR,
  pattern = "\\.(tif|tiff|png|jpg|jpeg)$",
  recursive = TRUE,
  full.names = TRUE,
  ignore.case = TRUE
)
image_index <- image_index[!grepl("\\.svs$", image_index, ignore.case = TRUE)]
image_keys <- tolower(tools::file_path_sans_ext(basename(image_index)))
image_lookup <- split(image_index, image_keys)

find_image_for_xml <- function(xml_path) {
  key <- tolower(tools::file_path_sans_ext(basename(xml_path)))
  candidates <- image_lookup[[key]]
  if (is.null(candidates) || !length(candidates)) {
    return(NA_character_)
  }

  # Prefer TIF/TIFF files; they are the matching image format in this dataset.
  ext <- tolower(tools::file_ext(candidates))
  preferred <- candidates[ext %in% c("tif", "tiff")]
  if (length(preferred)) return(preferred[1])
  candidates[1]
}

read_rgb <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext == "svs") {
    stop("SVS whole-slide files are not read by this script; use the matching TIF/TIFF image.")
  }

  im <- EBImage::readImage(path)
  a <- EBImage::imageData(im)
  d <- dim(a)

  if (is.null(d) || length(d) < 2L || any(!is.finite(d[1:2])) ||
      any(d[1:2] < 1L)) {
    stop("Image has invalid dimensions: ", paste(d, collapse = " x "))
  }

  if (length(d) > 3L) {
    idx <- rep(list(TRUE), length(d))
    idx[[length(d)]] <- 1L
    a <- do.call(`[`, c(list(a), idx, list(drop = TRUE)))
    d <- dim(a)
  }

  if (length(d) == 2L) {
    a <- array(rep(a, 3L), dim = c(d[1], d[2], 3L))
  } else if (length(d) == 3L) {
    if (d[3] == 1L) {
      one <- a[, , 1L]
      a <- array(rep(one, 3L), dim = c(d[1], d[2], 3L))
    } else if (d[3] >= 3L) {
      a <- a[, , 1:3, drop = FALSE]
    } else {
      stop("Unsupported image channel count: ", d[3])
    }
  } else {
    stop("Unsupported image dimensions: ", paste(d, collapse = " x "))
  }

  # EBImage stores x, y, channel; convert to y, x, channel for mask alignment.
  a <- aperm(a, c(2, 1, 3))

  # Clip pixel values in place so array dimensions are preserved.
  a[a < 0] <- 0
  a[a > 1] <- 1

  h <- dim(a)[1]
  w <- dim(a)[2]

  if (!is.numeric(w) || !is.numeric(h) || length(w) != 1L ||
      length(h) != 1L || is.na(w) || is.na(h) || w < 1L || h < 1L) {
    stop("Could not determine numeric image width/height.")
  }

  list(array = a, width = as.integer(w), height = as.integer(h))
}

polygon_to_mask <- function(xs, ys, width, height) {
  mask <- matrix(FALSE, nrow = height, ncol = width)
  n <- length(xs)
  if (n < 3 || any(!is.finite(xs)) || any(!is.finite(ys))) return(mask)

  xs <- pmax(1, pmin(width, xs))
  ys <- pmax(1, pmin(height, ys))
  ymin <- max(1L, floor(min(ys)))
  ymax <- min(height, ceiling(max(ys)))
  if (ymin > ymax) return(mask)

  for (row in seq.int(ymin, ymax)) {
    y <- row
    intersections <- numeric(0)
    for (k in seq_len(n)) {
      k2 <- if (k == n) 1L else k + 1L
      x1 <- xs[k]; y1 <- ys[k]
      x2 <- xs[k2]; y2 <- ys[k2]
      if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
        intersections <- c(intersections, x1 + (y - y1) * (x2 - x1) / (y2 - y1))
      }
    }
    if (length(intersections) >= 2) {
      intersections <- sort(intersections)
      starts <- seq.int(1L, length(intersections) - 1L, by = 2L)
      for (j in starts) {
        left <- max(1L, ceiling(intersections[j]))
        right <- min(width, floor(intersections[j + 1L]))
        if (left <= right) mask[row, left:right] <- TRUE
      }
    }
  }
  mask
}

get_class_name <- function(annotation_node) {
  attr_nodes <- xml2::xml_find_all(annotation_node, ".//Attribute")
  candidates <- c(
    xml2::xml_attr(attr_nodes, "Name"),
    xml2::xml_attr(annotation_node, "Name"),
    xml2::xml_attr(annotation_node, "PartOfGroup")
  )
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  if (!length(candidates)) return("unknown")
  val <- tolower(trimws(candidates[1]))
  val <- gsub("[ _-]+", "", val)
  if (grepl("epithelial", val)) return("epithelial")
  if (grepl("lymphocyte", val)) return("lymphocyte")
  if (grepl("neutrophil", val)) return("neutrophil")
  if (grepl("macrophage", val)) return("macrophage")
  if (grepl("ambiguous|unknown|unclassified", val)) return("ignore")
  "unknown"
}

make_boundary <- function(instance_mask, thickness = 2L) {
  boundary <- matrix(0L, nrow = nrow(instance_mask), ncol = ncol(instance_mask))
  ids <- sort(unique(as.vector(instance_mask)))
  ids <- ids[ids > 0]
  if (!length(ids)) return(boundary)

  for (id in ids) {
    obj <- instance_mask == id
    eroded <- obj
    for (k in seq_len(thickness)) {
      padded <- matrix(FALSE, nrow(eroded) + 2L, ncol(eroded) + 2L)
      padded[2:(nrow(padded)-1L), 2:(ncol(padded)-1L)] <- eroded
      eroded <- padded[1:(nrow(padded)-2L), 2:(ncol(padded)-1L)] &
        padded[3:nrow(padded), 2:(ncol(padded)-1L)] &
        padded[2:(nrow(padded)-1L), 1:(ncol(padded)-2L)] &
        padded[2:(nrow(padded)-1L), 3:ncol(padded)] &
        padded[2:(nrow(padded)-1L), 2:(ncol(padded)-1L)]
    }
    boundary[obj & !eroded] <- 1L
  }
  boundary
}

nucleus_stats <- function(instance_mask, class_mask, class_lookup, image_id, patient_id) {
  ids <- sort(unique(as.vector(instance_mask)))
  ids <- ids[ids > 0]
  if (!length(ids)) return(tibble())

  map_dfr(ids, function(id) {
    idx <- which(instance_mask == id, arr.ind = TRUE)
    ys <- idx[, 1]; xs <- idx[, 2]
    w <- max(xs) - min(xs) + 1
    h <- max(ys) - min(ys) + 1
    cl <- class_lookup[[as.character(id)]]
    if (is.null(cl) || !length(cl)) cl <- "unknown"
    tibble(
      patient_id = patient_id,
      image_id = image_id,
      instance_id = as.integer(id),
      class = as.character(cl),
      area_px = length(xs),
      equivalent_diameter_px = 2 * sqrt(length(xs) / pi),
      aspect_ratio = max(w, h) / max(1, min(w, h)),
      centroid_x = mean(xs),
      centroid_y = mean(ys),
      bbox_width = w,
      bbox_height = h
    )
  })
}

image_panel <- function(rgb, instance_mask, class_mask, boundary_mask, out_path) {
  png(out_path, width = 1400, height = 420, res = 130)
  oldpar <- par(no.readonly = TRUE)
  on.exit({par(oldpar); dev.off()}, add = TRUE)
  par(mfrow = c(1, 4), mar = c(1, 1, 3, 1))
  display_rgb <- rgb
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "", main = "Input image")
  rasterImage(as.raster(display_rgb), 0, 0, 1, 1)
  plot(NA, xlim = c(1, ncol(instance_mask)), ylim = c(nrow(instance_mask), 1),
       axes = FALSE, xlab = "", ylab = "", main = "Instance IDs")
  rasterImage(as.raster(instance_mask > 0), 1, nrow(instance_mask), ncol(instance_mask), 1)
  plot(NA, xlim = c(1, ncol(class_mask)), ylim = c(nrow(class_mask), 1),
       axes = FALSE, xlab = "", ylab = "", main = "Class mask")
  rasterImage(as.raster(class_mask / max(1, max(class_mask))), 1, nrow(class_mask), ncol(class_mask), 1)
  plot(NA, xlim = c(1, ncol(boundary_mask)), ylim = c(nrow(boundary_mask), 1),
       axes = FALSE, xlab = "", ylab = "", main = "Boundary target")
  rasterImage(as.raster(boundary_mask), 1, nrow(boundary_mask), ncol(boundary_mask), 1)
}

xml_files <- list.files(DATA_DIR, pattern = "\\.xml$", recursive = TRUE, full.names = TRUE,
                         ignore.case = TRUE)
if (!length(xml_files)) {
  stop("No XML annotations found. Check DATA_DIR.")
}
cat("XML annotation files found:", length(xml_files), "\\n")
cat("TIF/TIFF and other supported image files found:", length(image_index), "\\n")
matched_xml <- vapply(xml_files, function(x) !is.na(find_image_for_xml(x)), logical(1))
cat("XML files with matching images:", sum(matched_xml), "\\n")
cat("XML files without matching images:", sum(!matched_xml), "\\n")
if (is.finite(MAX_IMAGES)) xml_files <- head(xml_files, MAX_IMAGES)

image_rows <- list()
nucleus_rows <- list()
split_rows <- list()
processed <- 0L

for (xml_path in xml_files) {
  image_path <- find_image_for_xml(xml_path)
  if (is.na(image_path)) {
    warn_skip("No matching image found for XML: %s", basename(xml_path))
    next
  }

  rec <- tryCatch(read_rgb(image_path), error = function(e) {
    warn_skip("Could not read image %s: %s", basename(image_path), conditionMessage(e))
    NULL
  })
  if (is.null(rec)) next

  doc <- tryCatch(xml2::read_xml(xml_path), error = function(e) {
    warn_skip("Could not parse XML %s: %s", basename(xml_path), conditionMessage(e))
    NULL
  })
  if (is.null(doc)) next

  width <- rec$width
  height <- rec$height
  if (length(width) != 1L || length(height) != 1L ||
      !is.finite(width) || !is.finite(height) || width < 1L || height < 1L) {
    warn_skip("Invalid dimensions for %s; skipping this image.", basename(image_path))
    next
  }
  instance_mask <- matrix(0L, nrow = as.integer(height), ncol = as.integer(width))
  class_mask <- matrix(0L, nrow = height, ncol = width)
  class_lookup <- list()
  instance_id <- 0L

  anns <- xml2::xml_find_all(doc, ".//Annotation")
  if (!length(anns)) anns <- list(doc)

  for (ann in anns) {
    cl <- get_class_name(ann)
    regions <- xml2::xml_find_all(ann, ".//Region")
    if (!length(regions)) next

    for (region in regions) {
      verts <- xml2::xml_find_all(region, ".//Vertex")
      if (length(verts) < 3) next
      xs <- suppressWarnings(as.numeric(xml2::xml_attr(verts, "X")))
      ys <- suppressWarnings(as.numeric(xml2::xml_attr(verts, "Y")))
      good <- is.finite(xs) & is.finite(ys)
      xs <- xs[good]; ys <- ys[good]
      if (length(xs) < 3) next

      pm <- polygon_to_mask(xs, ys, width, height)
      pm <- pm & (instance_mask == 0L)
      if (!any(pm)) next

      instance_id <- instance_id + 1L
      instance_mask[pm] <- instance_id
      class_lookup[[as.character(instance_id)]] <- cl

      if (cl == "ignore" || cl == "unknown" || !(cl %in% classes)) {
        class_mask[pm] <- 255L
      } else {
        class_mask[pm] <- class_id[[cl]]
      }
    }
  }

  if (instance_id == 0L) {
    warn_skip("No usable annotation polygons in %s", basename(xml_path))
    next
  }

  boundary_mask <- make_boundary(instance_mask, thickness = 2L)
  boundary_mask[class_mask == 255L] <- 255L

  base_image_id <- tools::file_path_sans_ext(basename(image_path))
  patient_id <- basename(dirname(xml_path))
  if (patient_id %in% c(".", basename(DATA_DIR))) patient_id <- basename(dirname(dirname(xml_path)))
  image_id <- paste(patient_id, base_image_id, sep = "__")

  stats <- nucleus_stats(instance_mask, class_mask, class_lookup, image_id, patient_id)
  if (!nrow(stats)) next

  rgb_file <- file.path(MASK_DIR, paste0(image_id, "_rgb.rds"))
  inst_file <- file.path(MASK_DIR, paste0(image_id, "_instance.rds"))
  cls_file <- file.path(MASK_DIR, paste0(image_id, "_class.rds"))
  bnd_file <- file.path(MASK_DIR, paste0(image_id, "_boundary.rds"))
  saveRDS(rec$array, rgb_file)
  saveRDS(instance_mask, inst_file)
  saveRDS(class_mask, cls_file)
  saveRDS(boundary_mask, bnd_file)

  panel_file <- file.path(PANEL_DIR, paste0(image_id, "_panel.png"))
  tryCatch(image_panel(rec$array, instance_mask, class_mask, boundary_mask, panel_file),
           error = function(e) warn_skip("Could not create panel for %s: %s", image_id, conditionMessage(e)))

  image_rows[[length(image_rows) + 1L]] <- tibble(
    patient_id = patient_id,
    image_id = image_id,
    image_path = normalizePath(image_path, winslash = "/", mustWork = FALSE),
    xml_path = normalizePath(xml_path, winslash = "/", mustWork = FALSE),
    image_width = width,
    image_height = height,
    n_nuclei = nrow(stats),
    mean_red = mean(rec$array[, , 1], na.rm = TRUE) * 255,
    mean_green = mean(rec$array[, , 2], na.rm = TRUE) * 255,
    mean_blue = mean(rec$array[, , 3], na.rm = TRUE) * 255,
    brightness = mean(rec$array, na.rm = TRUE) * 255,
    foreground_coverage_pct = 100 * mean(instance_mask > 0)
  )
  nucleus_rows[[length(nucleus_rows) + 1L]] <- stats
  processed <- processed + 1L
  message(sprintf("Processed %d/%d: %s", processed, length(xml_files), image_id))
}

image_df <- bind_rows(image_rows)
nucleus_df <- bind_rows(nucleus_rows)

if (!nrow(image_df) || !nrow(nucleus_df)) {
  stop("No images with usable annotations were processed. Check folder structure and XML format.")
}

patient_ids <- unique(image_df$patient_id)
set.seed(42)
patient_ids <- sample(patient_ids)
n_pat <- length(patient_ids)
n_train <- max(1L, floor(0.70 * n_pat))
n_val <- max(1L, floor(0.15 * n_pat))
if (n_train + n_val >= n_pat) n_val <- max(1L, n_pat - n_train - 1L)
train_patients <- patient_ids[seq_len(n_train)]
val_patients <- patient_ids[seq.int(n_train + 1L, n_train + n_val)]
test_patients <- setdiff(patient_ids, c(train_patients, val_patients))

image_df <- image_df %>%
  mutate(split = case_when(
    patient_id %in% train_patients ~ "train",
    patient_id %in% val_patients ~ "validation",
    TRUE ~ "test"
  ))
nucleus_df <- nucleus_df %>%
  left_join(image_df %>% select(patient_id, image_id, split), by = c("patient_id", "image_id"))

write.csv(image_df, file.path(OUT_DIR, "image_level_statistics.csv"), row.names = FALSE)
write.csv(nucleus_df, file.path(OUT_DIR, "nucleus_level_statistics.csv"), row.names = FALSE)
write.csv(image_df %>% select(patient_id, image_id, split), file.path(OUT_DIR, "patient_level_split.csv"), row.names = FALSE)

split_df <- image_df %>% count(split, name = "images") %>%
  left_join(image_df %>% distinct(patient_id, split) %>% count(split, name = "patients"), by = "split")
p_split <- ggplot(split_df, aes(split, images)) +
  geom_col(fill = "#457B9D") +
  geom_text(aes(label = images), vjust = -0.3) +
  labs(title = "Actual MoNuSAC images per patient-level split", x = "Split", y = "Image count")
save_plot(p_split, "01_images_per_split.png")

p_patients <- ggplot(split_df, aes(split, patients)) +
  geom_col(fill = "#2A9D8F") +
  geom_text(aes(label = patients), vjust = -0.3) +
  labs(title = "Actual patients per split", x = "Split", y = "Patient count")
save_plot(p_patients, "02_patients_per_split.png")

class_counts <- nucleus_df %>% filter(class %in% classes) %>%
  count(class, name = "nuclei") %>%
  complete(class = classes, fill = list(nuclei = 0))
class_counts$class <- factor(class_counts$class, levels = classes)
p_class <- ggplot(class_counts, aes(class, nuclei, fill = class)) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = class_cols) +
  geom_text(aes(label = scales::comma(nuclei)), vjust = -0.3) +
  labs(title = "Annotated nuclei per class (actual XML annotations)", x = NULL, y = "Nucleus count")
save_plot(p_class, "03_nuclei_count_by_class.png")

p_area <- ggplot(nucleus_df %>% filter(class %in% classes, area_px > 0),
                 aes(class, area_px, fill = class)) +
  geom_violin(trim = TRUE, show.legend = FALSE) +
  geom_boxplot(width = 0.12, outlier.alpha = 0.15) +
  scale_y_log10() + scale_fill_manual(values = class_cols) +
  labs(title = "Nucleus area by class (actual annotations)", x = "Class", y = "Area (pixels, log scale)")
save_plot(p_area, "04_nucleus_area_by_class.png")

p_diam <- ggplot(nucleus_df %>% filter(class %in% classes),
                 aes(equivalent_diameter_px, colour = class, fill = class)) +
  geom_density(alpha = 0.12) + scale_colour_manual(values = class_cols) +
  scale_fill_manual(values = class_cols) +
  labs(title = "Equivalent nucleus diameter by class", x = "Equivalent diameter (pixels)", y = "Density")
save_plot(p_diam, "05_equivalent_diameter_by_class.png")

p_aspect <- ggplot(nucleus_df %>% filter(class %in% classes),
                   aes(class, aspect_ratio, fill = class)) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.15) +
  scale_fill_manual(values = class_cols) +
  labs(title = "Nucleus aspect ratio by class", x = "Class", y = "Aspect ratio")
save_plot(p_aspect, "06_aspect_ratio_by_class.png")

p_count <- ggplot(image_df, aes(n_nuclei, fill = split)) +
  geom_histogram(bins = 25, colour = "white") +
  labs(title = "Annotated nuclei per image", x = "Nuclei per image", y = "Image count", fill = "Split")
save_plot(p_count, "07_nuclei_per_image.png")

p_size <- ggplot(image_df, aes(image_width, image_height, colour = split)) +
  geom_point(size = 2.2, alpha = 0.8) +
  labs(title = "Actual image dimensions", x = "Width (pixels)", y = "Height (pixels)", colour = "Split")
save_plot(p_size, "08_image_sizes.png")

p_stain <- ggplot(image_df, aes(mean_red, mean_blue, colour = split)) +
  geom_point(size = 2.3, alpha = 0.8) +
  labs(title = "Per-image mean red vs blue intensity", x = "Mean red intensity (0-255)",
       y = "Mean blue intensity (0-255)", colour = "Split")
save_plot(p_stain, "09_stain_variability.png")

p_brightness <- ggplot(image_df, aes(brightness, fill = split, colour = split)) +
  geom_density(alpha = 0.15) +
  labs(title = "Image brightness distribution", x = "Mean RGB intensity (0-255)", y = "Density")
save_plot(p_brightness, "10_brightness_distribution.png")

p_coverage <- ggplot(image_df, aes(foreground_coverage_pct, fill = split)) +
  geom_histogram(bins = 25, colour = "white") +
  labs(title = "Annotated nucleus coverage per image", x = "Annotated area (%)", y = "Image count", fill = "Split")
save_plot(p_coverage, "11_foreground_coverage.png")

class_by_image <- nucleus_df %>% filter(class %in% classes) %>%
  count(image_id, class, name = "n") %>%
  complete(image_id = image_df$image_id, class = classes, fill = list(n = 0)) %>%
  left_join(image_df %>% select(image_id, split), by = "image_id")
heat_df <- class_by_image
p_heat <- ggplot(heat_df, aes(class, reorder(image_id, n), fill = log1p(n))) +
  geom_tile() +
  scale_fill_viridis_c(name = "log(1 + count)") +
  labs(title = "Class composition per image", x = "Nucleus class", y = "Image")
save_plot(p_heat, "12_class_composition_heatmap.png", width = 8, height = 10)

summary_tbl <- image_df %>%
  summarise(
    n_images = n(),
    n_patients = n_distinct(patient_id),
    total_annotated_instances = sum(n_nuclei),
    mean_nuclei_per_image = mean(n_nuclei),
    mean_width = mean(image_width),
    mean_height = mean(image_height)
  )
write.csv(summary_tbl, file.path(OUT_DIR, "dataset_summary.csv"), row.names = FALSE)
print(summary_tbl)
print(class_counts)

# Native-resolution preprocessing: instance, class and boundary masks are saved as RDS.
# Create native 256x256 crops; labels outside the image are padded with ignore value 255.
pad_crop <- function(arr, x0, y0, size, is_mask = FALSE, pad_value = 0) {
  if (is.matrix(arr)) {
    h <- nrow(arr); w <- ncol(arr)
    out <- matrix(pad_value, nrow = size, ncol = size)
    x1 <- max(1L, x0); y1 <- max(1L, y0)
    x2 <- min(w, x0 + size - 1L); y2 <- min(h, y0 + size - 1L)
    if (x1 <= x2 && y1 <= y2) {
      out[(y1-y0+1L):(y2-y0+1L), (x1-x0+1L):(x2-x0+1L)] <- arr[y1:y2, x1:x2, drop = FALSE]
    }
    return(out)
  }
  h <- dim(arr)[1]; w <- dim(arr)[2]; ch <- dim(arr)[3]
  out <- array(0, dim = c(size, size, ch))
  x1 <- max(1L, x0); y1 <- max(1L, y0)
  x2 <- min(w, x0 + size - 1L); y2 <- min(h, y0 + size - 1L)
  if (x1 <= x2 && y1 <= y2) {
    out[(y1-y0+1L):(y2-y0+1L), (x1-x0+1L):(x2-x0+1L), ] <- arr[y1:y2, x1:x2, , drop = FALSE]
  }
  out
}

for (i in seq_len(nrow(image_df))) {
  row <- image_df[i, ]
  image_id <- row$image_id
  rgb <- readRDS(file.path(MASK_DIR, paste0(image_id, "_rgb.rds")))
  inst <- readRDS(file.path(MASK_DIR, paste0(image_id, "_instance.rds")))
  cls <- readRDS(file.path(MASK_DIR, paste0(image_id, "_class.rds")))
  bnd <- readRDS(file.path(MASK_DIR, paste0(image_id, "_boundary.rds")))

  ids <- which(inst > 0, arr.ind = TRUE)
  if (!nrow(ids)) next

  for (k in seq_len(4L)) {
    if (runif(1) < 0.60) {
      point <- ids[sample(seq_len(nrow(ids)), 1), ]
      cx <- point[2]; cy <- point[1]
      x0 <- as.integer(cx - CROP_SIZE %/% 2L)
      y0 <- as.integer(cy - CROP_SIZE %/% 2L)
    } else {
      x0 <- sample(seq_len(max(1L, row$image_width)), 1)
      y0 <- sample(seq_len(max(1L, row$image_height)), 1)
    }
    rgb_crop <- pad_crop(rgb, x0, y0, CROP_SIZE, pad_value = 0)
    inst_crop <- pad_crop(inst, x0, y0, CROP_SIZE, pad_value = 255L)
    cls_crop <- pad_crop(cls, x0, y0, CROP_SIZE, pad_value = 255L)
    bnd_crop <- pad_crop(bnd, x0, y0, CROP_SIZE, pad_value = 255L)

    crop_name <- paste0(gsub("[^A-Za-z0-9_-]", "_", image_id), "_crop_", k, ".rds")
    imagenet_mean <- c(0.485, 0.456, 0.406)
    imagenet_std <- c(0.229, 0.224, 0.225)
    rgb_normalized <- rgb_crop
    for (ch in 1:3) {
      rgb_normalized[, , ch] <- (rgb_crop[, , ch] - imagenet_mean[ch]) / imagenet_std[ch]
    }
    saveRDS(list(image = rgb_crop, image_normalized = rgb_normalized,
                 instance_mask = inst_crop, class_mask = cls_crop,
                 boundary_mask = bnd_crop,
                 patient_id = row$patient_id, split = row$split),
            file.path(CROP_DIR, crop_name))
  }
}

cat("\nFinished preprocessing and EDA.\n")
cat("Processed images:", nrow(image_df), "\n")
cat("Output folder:", normalizePath(OUT_DIR, winslash = "/", mustWork = FALSE), "\n")
cat("Important: this script creates data-derived EDA plots and annotation targets.\n")
cat("It does not train Attention UNet++ or generate model predictions/attention maps/training curves.\n")
