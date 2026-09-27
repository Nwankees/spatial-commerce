plugins {
    id("com.android.application")
}

android {
    namespace = "com.hackgt.spatialcommerce"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.hackgt.spatialcommerce"
        minSdk = 24
        targetSdk = 37
        versionCode = 5
        versionName = "0.5.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("com.google.ar:core:1.56.0")
    // Remote product thumbnails (caching, downsampling, cancellation).
    implementation("io.coil-kt.coil3:coil:3.3.0")
    implementation("io.coil-kt.coil3:coil-network-okhttp:3.3.0")

    // JVM unit tests for the deterministic fit engine.
    testImplementation("junit:junit:4.13.2")
}
