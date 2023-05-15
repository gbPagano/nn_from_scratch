use polars::prelude::*;

fn main() {
    // Caminho do arquivo CSV
    let file_path = "train.csv";

    // Ler o arquivo CSV para um DataFrame
    let df = CsvReader::from_path(file_path)
        .unwrap()
        .infer_schema(None)
        .has_header(true)
        .finish()
        .unwrap();

    // Exibir o DataFrame
    println!("{:?}", df);
}