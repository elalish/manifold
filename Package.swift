// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "manifold",
    platforms: [
        .macOS(.v11), .iOS(.v13)
    ],
    products: [
        .library(
            name: "manifold",
            targets: ["manifold"])
    ],
    dependencies: [
        .package(url: "https://github.com/audulus/glm", branch: "spm"),
        .package(url: "https://github.com/audulus/Clipper2", branch: "spm"),
        .package(url: "https://github.com/audulus/tbb-spm", branch: "main"),
    ],
    targets: [
        .target(
            name: "manifold",
            dependencies: ["glm", "Clipper2", .product(name: "tbb", package: "tbb-spm")],
            path: "src",
            publicHeadersPath: "include",
            cxxSettings: [
                .define("MANIFOLD_PAR", to: "'T'"),
            ]
        )
    ],
    cxxLanguageStandard: .cxx20
)
