//! A load-dynamic build whose ONNX Runtime library cannot be loaded reports
//! a failed extractor instead of panicking. Its own test binary: ort reads
//! ORT_DYLIB_PATH once per process.
#![cfg(feature = "gliner-load-dynamic")]

use lemmalog::agent::{Episode, Extractor};
use lemmalog::extract::setup::{build_gliner, GlinerConfig};
use lemmalog::extract::LazyExtractor;

#[test]
fn missing_onnx_runtime_is_a_failed_load_not_a_crash() {
    let dir = std::env::temp_dir().join(format!("lemmalog-ort-test-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("onnx")).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    std::fs::write(dir.join("onnx/model.onnx"), b"not a model").unwrap();
    let cfg = GlinerConfig {
        tokenizer: dir.join("tokenizer.json").display().to_string(),
        model: dir.join("onnx/model.onnx").display().to_string(),
        glinker_pipeline: None,
        ort_dylib: Some("/nonexistent/libonnxruntime.so".into()),
    };
    // a named library that does not exist is reported before any load
    assert_eq!(cfg.missing(), vec!["/nonexistent/libonnxruntime.so".to_string()]);

    // forcing the load anyway: ort panics inside; the extractor reports it
    std::env::remove_var("ORT_DYLIB_PATH");
    let c = cfg.clone();
    let mut x = LazyExtractor::new("gliner: test", Box::new(move || build_gliner(&c)));
    let ep = Episode { id: "e".into(), text: "Bob works at Acme.".into(), ts: 0, speaker: None };
    assert!(x.extract(&ep).is_empty());
    let err = x.error().expect("load failed").to_string();
    assert!(err.contains("could not load ONNX Runtime"), "{err}");
    assert!(err.contains("/nonexistent/libonnxruntime.so"), "{err}");
    assert_eq!(x.stats(), (0, 1));
    let _ = std::fs::remove_dir_all(&dir);
}
