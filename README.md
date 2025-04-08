# Installing Apertium Kazakh Analyzer (`apertium-kaz`) on Windows using WSL

This guide provides step-by-step instructions to install the `apertium-kaz` morphological analyzer on a Windows machine using the Windows Subsystem for Linux (WSL), specifically compiling from source. This approach is necessary to overcome potential Windows filesystem limitations and allows for modification of the linguistic data.

## Prerequisites

* Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11.
* Administrator privileges on Windows.
* A stable internet connection.

## Step 1: Install WSL

1.  **Open PowerShell or Command Prompt as Administrator:**
    * Search for "PowerShell" or "cmd" in the Start menu.
    * Right-click on the result and select "Run as administrator".

2.  **Run the WSL Install Command:**
    ```powershell
    wsl --install
    ```
    * This command enables necessary Windows features, downloads the Linux kernel, sets WSL 2 as default, and installs the default Linux distribution (usually Ubuntu).
    * **Note:** This command is safe and will **not** delete your Windows installation or personal files.
    * If the installation seems stuck (e.g., at 0% downloading Ubuntu), check your internet connection, firewall/VPN settings, and ensure Windows Update is working. Sometimes a simple computer restart helps. If problems persist, consult the [official Microsoft WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).

3.  **Restart Your Computer:** A restart is usually required after the installation finishes to complete the setup.

## Step 2: Initial WSL/Ubuntu Setup

1.  **Launch Ubuntu:** After restarting, find "Ubuntu" (or your installed distribution) in the Start menu and launch it.
2.  **Create Linux User:** On the first launch, you will be prompted to create a username and password for your Linux environment.
    * Choose a simple username (e.g., lowercase, no spaces).
    * Choose a strong password (characters won't display while typing).
    * Remember these credentials – they are for Linux (`sudo`), not your Windows login.

## Step 3: Install Development Tools and Dependencies

Now, inside the Ubuntu terminal, install the necessary tools and libraries required to compile Apertium components.

1.  **Update Package Lists:**
    ```bash
    sudo apt update
    ```

2.  **Upgrade Existing Packages (Recommended):**
    ```bash
    sudo apt upgrade
    ```
    * Press `Y` to confirm if prompted. This keeps your WSL environment up-to-date.

3.  **Install Core Build Tools:**
    ```bash
    sudo apt install build-essential autoconf automake libtool pkg-config git
    ```
    * `build-essential`: Includes `make`, `gcc`, `g++` etc.
    * `autoconf`, `automake`, `libtool`: The Autotools build system.
    * `pkg-config`: Used by `configure` scripts to find libraries.
    * `git`: Needed to clone the repository.

4.  **Install Apertium Core Dependencies:**
    ```bash
    sudo apt install apertium lttoolbox apertium-dev lttoolbox-dev
    ```
    * `apertium`, `lttoolbox`: The core Apertium engine and tools.
    * `apertium-dev`, `lttoolbox-dev`: Development files needed to compile other modules *against* Apertium/lttoolbox (provides `.pc` files for `pkg-config`).

5.  **Install HFST Dependency:** `apertium-kaz` uses the Helsinki Finite-State Technology toolkit.
    ```bash
    sudo apt install hfst libhfst-dev
    ```
    * `hfst`: The HFST command-line tools (like `hfst-lexc`).
    * `libhfst-dev`: Development files for HFST.

## Step 4: Clone the `apertium-kaz` Repository

It's crucial to clone and work *within* the Linux filesystem provided by WSL (e.g., your home directory `~`) for best performance and compatibility. Avoid cloning directly onto your Windows drives via `/mnt/c/`.

1.  **Navigate to Home Directory:**
    ```bash
    cd ~
    ```

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/apertium/apertium-kaz.git](https://github.com/apertium/apertium-kaz.git)
    ```

3.  **Enter the Directory:**
    ```bash
    cd apertium-kaz
    ```

## Step 5: Compile `apertium-kaz`

Now that the dependencies are installed and you are in the repository directory *within WSL*, you can compile the language package.

1.  **Generate `configure` Script:**
    ```bash
    ./autogen.sh
    ```
    * This script uses Autotools to generate the `./configure` script. Ignore warnings about deprecated macros if they appear.

2.  **Run `configure` Script:**
    ```bash
    ./configure
    ```
    * This checks your system and prepares the Makefiles.
    * *(Optional)* If you want to install to a custom location later, you can add a prefix: `./configure --prefix=/path/to/your/install/location`

3.  **Compile:**
    ```bash
    make
    ```
    * This performs the actual compilation, building the transducer binaries (`.bin` files) etc. This might take some time. Ignore warnings unless an actual `Error` stops the process.

## Step 6: Test the Analyzer (Before Installation)

After `make` finishes successfully, you can test the morphological analyzer directly using the `lt-proc` tool and the compiled binary file.

1.  **Find the Analyzer Binary:** It's likely named `kaz.automorf.bin`. You can confirm with `ls *.bin`.
2.  **Run a Test:**
    ```bash
    echo "Сәлем!" | lt-proc kaz.automorf.bin
    ```
3.  **Expected Output:** You should see one or more morphological analyses for "Сәлем!", similar to:
    ```
    ^Сәлем/Сәлем<ij>/Сәлем<n><nom>/Сәлем<n><attr>/Сәлем<n><nom>+е<cop><aor><p3><pl>/Сәлем<n><nom>+е<cop><aor><p3><sg>$^!/!<sent>$
    ```

## Step 7: Install (Optional)

If you want to install `apertium-kaz` system-wide (or to the `--prefix` specified during `./configure`) so you can use the standard `apertium` command modes:

```bash
sudo make install

Note: You might need to run sudo ldconfig after make install if libraries were installed to a system location.
Step 8: Test Installed Mode (Optional)

If you ran make install, you should now be able to use the defined Apertium mode:
Bash

echo "Сәлем!" | apertium kaz-morph

Appendix: VS Code Integration (Recommended)

For a much smoother development experience, integrate VS Code with WSL:

    Install VS Code on Windows if you haven't already.
    Install the "WSL" extension from the VS Code Marketplace (published by Microsoft).
    Open your project:
        Open your WSL (Ubuntu) terminal.
        Navigate to your project folder (e.g., cd ~/apertium-kaz).
        Type code . and press Enter.
    Work: VS Code will open, connected to WSL. You can edit files directly (they are saved in Linux), and use the integrated VS Code terminal (Ctrl+`) to run all commands (git, make, lt-proc, etc.) directly in your WSL environment.

Appendix: Common Troubleshooting

    invalid path '...PRN...' during git clone: You are cloning onto the Windows filesystem (/mnt/c/...). Clone inside the Linux filesystem (~/) instead using WSL or a Linux VM.
    autoreconf: not found during ./autogen.sh: Core build tools are missing. Install them:
    Bash

sudo apt update
sudo apt install build-essential autoconf automake libtool pkg-config

checking for apertium >= ... no or Package 'apertium' ... not found during ./autogen.sh or ./configure: The Apertium core engine or its development files are missing or not detected by pkg-config. Install them:
Bash

sudo apt update
sudo apt install apertium lttoolbox apertium-dev lttoolbox-dev

Then verify with pkg-config --modversion apertium before running ./autogen.sh again.
hfst-lexc: not found during ./autogen.sh or ./configure: The HFST toolkit is missing. Install it:
Bash

sudo apt update
sudo apt install hfst libhfst-dev

WSL install stuck at 0%: Check internet, firewall, VPN, run wsl --install as Admin, check Windows Update, restart PC.
mv ... Permission denied when moving from /mnt/c/: Filesystem permission issue between Windows and WSL. Easiest fix: delete the folder on Windows (/mnt/c/...) using File Explorer and re-clone directly into WSL's filesystem (~/).
