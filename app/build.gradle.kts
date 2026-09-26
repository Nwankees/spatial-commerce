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
        versionCode = 3
        versionName = "0.3.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("com.google.ar:core:1.56.0")
}
