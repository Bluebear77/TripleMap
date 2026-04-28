import json
import os
import time
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Load OpenAI API credentials
with open("config/openai_credentials.txt") as f:
    sk = json.load(f)

# Initialize OpenAI client using the API key
client = OpenAI(api_key=sk["api_key"])

allowed_predicates = {
    "P2283": {
        "label": "uses",
        "description": "item or concept used by the subject or in the operation (see also instrument [P1303] and armament [P520])"
    },
    "P2144": {
        "label": "frequency",
        "description": "frequency in Hz at which the subject works, for example the frequency a radio station can be received"
    },
    "P880": {
        "label": "CPU",
        "description": "central processing unit found within the subject item"
    },
    "P6711": {
        "label": "data transfer speed",
        "description": "transfer speed through a bus or a communication medium"
    },
    "P2928": {
        "label": "storage capacity",
        "description": "data storage size of a device"
    },
    "P2284": {
        "label": "price",
        "description": "published price listed or paid for a product (use with unit of currency)"
    },
    "P2560": {
        "label": "GPU",
        "description": "graphics processing unit within a system"
    },
    "P5204": {
        "label": "commercialization date",
        "description": "date when an invention was first commercialized"
    },
    "P749": {
        "label": "parent organization or unit",
        "description": "parent organization or unit of an organization or unit, opposite of child organization or unit (P355); use instance of (P31) to distinguish organization (Q43229) and organization unit (Q10387680)"
    },
    "P31": {
        "label": "instance of",
        "description": "type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain"
    },
    "P2109": {
        "label": "nominal power output",
        "description": "power produced by the engine, plant, or reactor (use with unit of power)"
    },
    "P2669": {
        "label": "discontinuation date",
        "description": "date that the availability of a product or service was discontinued; see also 'dissolved, abolished or demolished' (P576) and 'service retirement' (P730) for pieces or classes of equipment"
    },
    "P1343": {
        "label": "described by source",
        "description": "work where this item is described"
    },
    "P9767": {
        "label": "edition/version",
        "description": "the official edition or version name of a resource; this name is not considered part of the title or subtitle of the resource"
    },
    "P2049": {
        "label": "width",
        "description": "width of an object"
    },
    "P2664": {
        "label": "units sold",
        "description": "sales figures of an object"
    },
    "P2139": {
        "label": "total revenue",
        "description": "income gained by an organization during a given time frame. Not to be confused with fiscal revenue"
    },
    "P2295": {
        "label": "net profit",
        "description": "private entity profit after accounting for all costs"
    },
    "P2652": {
        "label": "partnership with",
        "description": "partnership (commercial or/and non-commercial) between this organization and another organization or institution"
    },
    "P1128": {
        "label": "employees",
        "description": "total number of employees of a company at a given 'point in time' (P585). Most recent data would generally have preferred rank; data for previous years normal rank (not deprecated rank). Add data for recent years, don't overwrite"
    },
    "P169": {
        "label": "chief executive officer",
        "description": "highest-ranking corporate officer appointed as the CEO within an organization"
    },
    "P12323": {
        "label": "working memory type",
        "description": "specifies the type of working memory of this data object"
    },
    "P1056": {
        "label": "product or material produced",
        "description": "material or product, including services, produced or provided by an organization, industry, facility, or process"
    },
    "P2149": {
        "label": "clock frequency",
        "description": "frequency of the clock signal that synchronizes the operation of a processor or other digital circuit, measured in hertz (Hz)"
    },
    "P516": {
        "label": "powered by",
        "description": "equipment or engine used by the subject to convert a source or energy into mechanical energy"
    },
    "P306": {
        "label": "operating system",
        "description": "operating system (OS) on which a software works or the OS installed on hardware"
    },
    "P178": {
        "label": "developer",
        "description": "organization or person that developed the item"
    },
    "P2048": {
        "label": "height",
        "description": "vertical length of an entity"
    },
    "P2043": {
        "label": "length",
        "description": "measured dimension of an object"
    },
    "P2067": {
        "label": "mass",
        "description": "mass (in colloquial usage also known as weight) of the item"
    },
    "P5524": {
        "label": "horizontal depth",
        "description": "spatial extent for 3D object along the third axis, what is most commonly referred to as its depth. Compare with 'vertical depth' (P4511)"
    },
    "P8470": {
        "label": "order number",
        "description": "order number of a product or service"
    },
    "P13351": {
        "label": "model number",
        "description": "identifier assigned by the manufacturer to a specific product model, used to distinguish it from other variants or versions"
    },
    "P13525": {
        "label": "RAM capacity",
        "description": "amount of random-access memory (RAM) available in a device or system (use with unit of data size)"
    },
    "P2403": {
        "label": "total assets",
        "description": "total value of all resources owned or controlled by an organization at a given point in time, as reported on a balance sheet"
    }
}


example_table_1 = {
        "Surface Studio Configuration Options Microsoft Surface Studio tech specs Price Tier (USD)": ["2,999", "3,499", "4,199"],
        "Surface Studio Configuration Options Microsoft Surface Studio tech specs CPU": ["Intel Core i5-6440HQ (2.6 to 3.5 GHz)", "Intel Core i7-6820HQ (2.7 to 3.6 GHz)", "Intel Core i7-6820HQ (2.7 to 3.6 GHz)"],
        "Surface Studio Configuration Options Microsoft Surface Studio tech specs Integrated GPU": ["GTX 965M", "GTX 965M", "GTX 980M"],
        "Surface Studio Configuration Options Microsoft Surface Studio tech specs RAM": ["8 GB", "16 GB", "32 GB"],
        "Surface Studio Configuration Options Microsoft Surface Studio tech specs Internal Storage": ["1 TB SATA II HDD + 64GB SATA SSD", "1 TB SATA II HDD + 128GB NVMe SSD", "2 TB SATA III HDD + 128GB NVMe SSD"]
      }

example_text_1 = (
    "Hardware: The Surface Studio has a 28-inch 4.5K \"PixelSense\" display with 4500 x 3000 pixels, equivalent to 192 dpi. The screen, the thinnest ever built for an all-in-one PC at 12.5 millimetres thick, is capable of being used in both the DCI-P3 and sRGB color spaces, and features a unique hinge design that allows it be tilted to a flat position, in a manner similar to the Wacom Cintiq. The bezel of the display contains a 5.0 megapixel camera and a Windows Hello-compatible backlit infrared camera.\n\nThe CPU is located in the base. Its compact design contains a 6th generation (codename \"Skylake\") Intel Core i5 or Core i7 processor and either a NVIDIA GeForce GTX 965M or GeForce GTX 980M graphics processor (both dependent on configuration). The system can be configured with up to 32 GB of DDR4 RAM and a 2 terabyte hard drive. It also features four USB 3.0 ports, a Mini DisplayPort, an SDXC card reader and a headset connection.\n\nUnlike many desktop PCs, the Surface Studio supports Microsoft's Modern Standby (formerly known as InstantGo) specification, enabling background tasks to operate while the computer is sleeping. A firmware update was released in April 2017 that enabled Cortana to be summoned via a \"Hey, Cortana\" voice command from sleep, provided the Studio is running the Creators Update.",
   )

example_mapping_1 = {
  "Surface Studio Configuration Options Microsoft Surface Studio tech specs Internal Storage": "P2928",
  "Surface Studio Configuration Options Microsoft Surface Studio tech specs CPU": "P880",
  "Surface Studio Configuration Options Microsoft Surface Studio tech specs Integrated GPU": "P2560",
  "Surface Studio Configuration Options Microsoft Surface Studio tech specs RAM": "P13525",
  "Surface Studio Configuration Options Microsoft Surface Studio tech specs Price Tier (USD)": "P2284"
}

example_text_2 = (
    "WaveLAN: WaveLAN was a brand name for a family of wireless networking technology sold by NCR, AT&T, Lucent Technologies, and Agere Systems as well as being sold by other companies under OEM agreements. The WaveLAN name debuted on the market in 1990 and was in use until 2000, when Agere Systems renamed their products to ORiNOCO. WaveLAN laid the important foundation for the formation of IEEE 802.11 working group and the resultant creation of Wi-Fi.\n\nWaveLAN has been used on two different families of wireless technology:\n* Pre-IEEE 802.11 WaveLAN, also called Classic WaveLAN\n* IEEE 802.11-compliant WaveLAN, also known as WaveLAN IEEE and ORiNOCO History: WaveLAN was originally designed by NCR Systems Engineering, later renamed into WCND (Wireless Communication and Networking Division) at Nieuwegein, in the province Utrecht in the Netherlands, a subsidiary of NCR Corporation, in 1986–7, and introduced to the market in 1990 as a wireless alternative to Ethernet and Token Ring. The next year NCR contributed the WaveLAN design to the IEEE 802 LAN/MAN Standards Committee. This led to the founding of the 802.11 Wireless LAN Working Committee which produced the original IEEE 802.11 standard, which eventually became the basis of the certification mark Wi-Fi. When NCR was acquired by AT&T in 1991, becoming the AT&T GIS (Global Information Solutions) business unit, the product name was retained, as happened two years later when the product was transferred to the AT&T GBCS (Global Business Communications Systems) business unit, and again when AT&T spun off their GBCS business unit as Lucent in 1995. The technology was also sold as WaveLAN under an OEM agreement by Epson, Hitachi, and NEC, and as the RoamAbout DS by DEC.  It competed directly with Aironet's non-802.11 ARLAN lineup, which offered similar speeds, frequency ranges and hardware.\n\nSeveral companies also marketed wireless bridges and routers based on the WaveLAN ISA and PC cards, like the C-Spec OverLAN, KarlNet KarlBridge, Persoft Intersect Remote Bridge, and Solectek AIRLAN/Bridge Plus.  Lucent's WavePoint II access point could accommodate both the classic WaveLAN PC cards as well as the WaveLAN IEEE cards. Also, there were a number of compatible third-party products available to address niche markets such as: Digital Ocean's  Grouper, Manta, and Starfish offerings for the Apple Newton and Macintosh; Solectek's 915 MHz WaveLAN parallel port adapter; Microplex's M204 WaveLAN-compatible wireless print server; NEC's Japanese-market only C&C-Net 2.4 GHz adapter for the NEC-bus; Toshiba's Japanese-market only WaveCOM 2.4 GHz adapter for the Toshiba-Bus; and Teklogix's WaveLAN-compatible Pen-based and Notebook terminals.\n\nDuring this time frame, networking professionals also realized that since NetWare 3.x and 4.x supported the WaveLAN cards and came with a Multi Protocol Router module that supported the IP/IPX RIP and OSPF routing protocols, one could construct a wireless routed network using NetWare servers and WaveLAN cards for a fraction of the cost of building a wireless bridged network using WaveLAN access points. Many NetWare classes and textbooks of the time included a NetWare OS CD with a 2-person license, so potentially the only cost incurred came from hardware.\n\nWhen the 802.11 protocol was ratified, Lucent began producing chipsets and PC-cards to support this new standard under the name of WaveLAN IEEE. WaveLAN was among the first products certified by the Wi-Fi Alliance, originally called the Wireless Ethernet Compatibility Association (WECA). Shortly thereafter, Lucent spun off its semiconductor  division that also produced the WaveLAN chipsets as Agere Systems. On June 17, 2002 Proxim acquired the IEEE 802.11 LAN equipment business including the trademark ORiNOCO from Agere Systems. Proxim later renamed its entire 802.11 wireless networking lineup to ORiNOCO, including products based on Atheros chipsets. Specifications: Classic WaveLAN operates in the 900 MHz or 2.4 GHz ISM bands. Being a proprietary pre-802.11 protocol, it is completely incompatible with the 802.11 standard. Soon after the publication of the IEEE 802.11 standard on November 18, 1997, WaveLAN IEEE was placed on the market. Hardware: The pre-802.11 standard WaveLAN cards were based on the Intel 82586 Ethernet PHY controller, which was a commonly used controller in its time and was found in many ISA and MCA Ethernet cards, such as the Intel EtherExpress 16 and the 3COM 3C523. The WaveLAN IEEE ISA, MCA and PCMCIA cards used Medium Access Controller (MAC), HERMES, designed specifically for 802.11 protocol support. The radio modem section was hidden from the OS, thus making the WaveLAN card appear to be a typical Ethernet card, with the radio-specific features taken care of behind the scenes.\n\nWhile the 900 MHz models and the early 2.4 GHz models operated on one fixed frequency, the later 2.4 GHz cards as well as some 2.4 GHz WavePoint access points had the hardware capacity to operate over ten channels, ranging from 2.412 GHz to 2.484 GHz, with the channels available being determined by the region-specific firmware. Security: For security, WaveLAN used a 16-bit NWID (NetWork ID) field, which yielded 65,536 potential combinations; the radio portion of the device could receive radio traffic tagged with another NWID, but the controller would discard the traffic.  DES encryption (56-bit) was an option in some of the ISA and MCA cards and all of the WavePoint access points. The full-length ISA and MCA cards had a socket for an encryption chip, the half-length 915 MHz ISA cards had solder pads for a socket which was never added, and the 2.4 GHz half-length ISA cards had the chip soldered directly to the board.\n\nFor the IEEE 802.11 standard the goal was to provide data confidentiality comparable to that of a traditional wired network, using 64- and 128-bit data encryption technology. This first implementation was called “Wired Equivalent Privacy” (WEP).\n\nThere are shortcomings in WaveLAN & initial 802.11 compatible devices security strategy: \n* The initial IEEE 802.11 security WEP implementation, was shown to be vulnerable to attack.   \nThis was addressed by the 802.11i Wi-Fi Protected Access (WPA) that replaced WEP in the standard. Official specifications:  Support:  Officially released drivers: *Windows 3.11, 95, and NT 3.5/4.0\n** Windows 3.11, Windows 95, and 98 supported the ISA and MCA cards natively but did not provide any configuration or link diagnostics utilities.\n**Windows NT 3.51 did not natively support the WaveLAN cards, but additional drivers from Microsoft's Windows NT Driver Library were available.\n*OS/2 NDIS and NetWare Requester\n*LAN Manager/IBM LAN Server\n*Artisoft LANtastic\n*PC-TCP for DOS\n*NetWare Lite, NetWare 2, 3, and 4.  Netware 4.11 through 5.x supported the ISA and MCA cards natively but did not provide any configuration or link diagnostics utilities.\n*ODI/VLM NetWare client for DOS. The DOS drivers came with configuration and link diagnostics utilities.\n*SCO UNIX version 1.00.00.00\n*UnixWare version 1.1\n*NCR's documentation stated that drivers for Banyan Vines 5.05 were available on Banyan's BBS, but it is unclear if they ever materialized Volunteer-developed drivers: Linux has included support for ISA Classic WaveLAN cards since the 2.0.37 kernel, while full support for the PC card Classic WaveLAN cards came later. Status of support for MCA Classic Wavelan cards is unknown.\n\nFreeBSD version 2.2.1-up and the Mach4 kernel have had native support for the ISA Classic WaveLAN cards for several years.  OpenBSD and NetBSD do not natively support any of the Classic WaveLAN cards.\n\nSeveral open-source projects, such as NdisWrapper and Project Evil, currently exist that allow the use of NDIS drivers via a \"wrapper\". This allows non-Windows OS' to utilize the near-universal nature of drivers written for the Windows platform to the benefit of other operating systems, such as Linux, FreeBSD, and ZETA. Examples: Classic WaveLAN technology was available for the MCA, ISA/EISA, and PCMCIA interfaces: 915 MHz: * Full-length ISA card\n** F connector\n** RG-59/U antenna cable\n** NCR 008-0126998 HOLI (HOst Lan Interface) chip\n**NCR 008-0126999 Icarus or NCR 008-0127211 Daedalus chip\n** Intel N82586 PHY controller chip\n** IRQ, boot ROM, and boot ROM base address configured with a four-position DIP switch block at top of card\n** NCR part number 601-0068991\n** AT&T part number 3399-F170\n* Half-length ISA card\n**SMB connector\n** NCR 008-0126998 HOLI chip\n**Intel N82586 PHY controller chip\n** IRQ, boot ROM, and boot ROM base address configured with a four-position DIP switch block at top of card\n**AT&T part number 3399-K602.\n*Full-length MCA card\n**F connector\n**NCR 008-0127216 HOLI chip\n**NCR 008-0126999 Icarus chip\n**NCR 8-127000A socketed DES encryption chip\n**Intel N82586 PHY controller chip\n**MCA id number 6A14.\n* PC card\n** Large EAM (External Antenna Module)\n** Intel i82593 PHY controller chip\n** AT&T part number 3399-K080\n** Compaq/DEC Roamabout part number: DEINA-AA. 2.4 GHz: * Full-length ISA card\n**Fixed frequency\n** IRQ, boot ROM, and boot ROM base address configured with a four-position DIP switch block at top of card\n* Half-length ISA card\n**SMB connector\n**Selectable frequency\n**Symbios Logic 008-0126998 HOLI chip\n** Intel N82586 PHY controller chip\n** IRQ, boot ROM, and boot ROM base address configured with a four-position DIP switch block at top of card\n**AT&T part number 3399-K635.\n*Full-length MCA card\n**SMB connector\n**NCR 008-0127216 HOLI chip\n**NCR 008-0127211 Daedalus chip\n**NCR 8-127000A socketed DES encryption chip\n**Intel N82586 PHY controller chip\n**AT&T part number 3399-K066\n**MCA id number 6A14.\n* PC card - 2.4 GHz, selectable frequency, large EAM (External Antenna Module).\n** Intel N82593 PHY controller chip\n**AT&T part number: AT&T 3399-K624.\n**Lucent part number: LUC 3399-K644.\n**Compaq/DEC Roamabout part number: DEIRB-xx. Options: * DES encryption chip.  Part number 3399-K972.\n* Boot ROM chip.  Part number 3399-K973. Citations:  References: * NCR WaveLAN PC-AT Installation and Operations manual, part number ST-2119-09, revision number 008-0127167 Rev. B, copyright 1990, 1991 by NCR Corporation. External links: *NCR's HTTP site with a selection of WaveLAN drivers and documentation\n*FTP mirror site of DEC's ftp server with a selection of RoamAbout drivers and documentation\n* Detailed analysis of WaveLAN ISA cards\n*Wayback machine archive of documentation on an NCR WaveLAN backbone built in Latvia\n* Wayback machine archive of Byte Magazine's review of WaveLAN\n*Wayback machine archive for Wavelan Classic products\n*Detailed analysis of Wavelan MCA cards\n\n\n\n\nCategory:Wireless networking\nCategory:Network access\nCategory:NCR Corporation products",
  )

example_table_2 = {
        "Realm": [
          "US & Canada",
          "Worldwide",
          "Europe (except France)",
          "France",
          "Australia",
          "Japan"
        ],
        "Type": [
          "900 MHz",
          "2.4 GHz",
          "2.4 GHz",
          "2.4 GHz",
          "2.4 GHz",
          "2.4 GHz"
        ],
        "Number of frequencies": [
          "1",
          "6",
          "8",
          "2",
          "4",
          "1"
        ],
        "Frequency": [
          "915 MHz",
          "2.412 GHz, 2.422 GHz, 2.432 GHz, 2.442 GHz, 2.452 GHz, 2.462 GHz",
          "2.422 GHz, 2.425 GHz, 2.4305 GHz, 2.432 GHz, 2.442 GHz, 2.452 GHz, 2.460 GHz, 2.462 GHz",
          "2.460 GHz and 2.462 GHz",
          "2.422 GHz, 2.425 GHz, 2.432 GHz, 2.442 GHz",
          "2.484 GHz"
        ],
        "Modulation technique": [
          "DSSS/DQPSK",
          "DSSS/DQPSK",
          "DSSS/DQPSK",
          "DSSS/DQPSK",
          "DSSS/DQPSK",
          "DSSS/DQPSK"
        ],
        "Output power": [
          "250 mW",
          "32 mW",
          "32 mW",
          "32 mW",
          "32 mW",
          "32 mW"
        ],
        "Maximum data rate": [
          "2 Mbit/s",
          "2 Mbit/s",
          "2 Mbit/s",
          "2 Mbit/s",
          "2 Mbit/s",
          "2 Mbit/s"
        ],
        "Media Access Control": [
          "CSMA/CA",
          "CSMA/CA",
          "CSMA/CA",
          "CSMA/CA",
          "CSMA/CA",
          "CSMA/CA"
        ],
        "Security": [
          "16-bit network ID and optional DES encryption",
          "16-bit network ID and optional DES encryption",
          "16-bit network ID and optional DES encryption",
          "16-bit network ID and optional DES encryption",
          "16-bit network ID and optional DES encryption",
          "16-bit network ID and optional DES encryption"
        ]
      }


example_mapping_2 = {
      "Frequency": "P2144",
      "Security": "P2283",
      "Output power": "P2109",
      "Maximum data rate": "P6711"
    }

example_text_3 = (
    "PCI Express (Peripheral Component Interconnect Express), officially abbreviated as PCIe or PCI-e, is a high-speed serial computer expansion bus standard, designed to replace the older PCI, PCI-X and AGP bus standards. It is the common motherboard interface for personal computers' graphics cards, sound cards, hard disk drive host adapters, SSDs, Wi-Fi and Ethernet hardware connections.  PCIe has numerous improvements over the older standards, including higher maximum system bus throughput, lower I/O pin count and smaller physical footprint, better performance scaling for bus devices, a more detailed error detection and reporting mechanism (Advanced Error Reporting, AER), and native hot-swap functionality. More recent revisions of the PCIe standard provide hardware support for I/O virtualization.\n\nThe PCI Express electrical interface is measured by the number of simultaneous lanes. (A lane is a single send/receive line of data. The analogy is a highway with traffic in both directions.) The interface is also used in a variety of other standards — most notably the laptop expansion card interface called ExpressCard. It is also used in the storage interfaces of SATA Express, U.2 (SFF-8639) and M.2.\n\nFormat specifications are maintained and developed by the PCI-SIG (PCI Special Interest Group) — a group of more than 900 companies that also maintains the conventional PCI specifications. The bonded serial bus architecture was chosen over the traditional parallel bus because of the inherent limitations of the latter, including half-duplex operation, excess signal count, and inherently lower bandwidth due to timing skew. Timing skew results from separate electrical signals within a parallel interface traveling through conductors of different lengths, on potentially different printed circuit board (PCB) layers, and at possibly different signal velocities.  Despite being transmitted simultaneously as a single word, signals on a parallel interface have different travel duration and arrive at their destinations at different times.  When the interface clock period is shorter than the largest time difference between signal arrivals, recovery of the transmitted word is no longer possible. Since timing skew over a parallel bus can amount to a few nanoseconds, the resulting bandwidth limitation is in the range of hundreds of megahertz.\n\n\n\nA serial interface does not exhibit timing skew because there is only one differential signal in each direction within each lane, and there is no external clock signal since clocking information is embedded within the serial signal itself. As such, typical bandwidth limitations on serial signals are in the multi-gigahertz range. PCI Express is one example of the general trend toward replacing parallel buses with serial interconnects; other examples include Serial ATA (SATA), USB, Serial Attached SCSI (SAS), FireWire (IEEE 1394), and RapidIO. In digital video, examples in common use are DVI, HDMI, and DisplayPort.\n\nMultichannel serial design increases flexibility with its ability to allocate fewer lanes for slower devices. Modern (since ) gaming video cards usually exceed the height as well as thickness specified in the PCI Express standard, due to the need for more capable and quieter cooling fans, as gaming video cards often emit hundreds of watts of heat. Modern computer cases are often wider to accommodate these taller cards, but not always. Since full-length cards (312 mm) are uncommon, modern cases sometimes cannot fit those. The thickness of these cards also typically occupies the space of 2 PCIe slots. In fact, even the methodology of how to measure the cards varies between vendors, with some including the metal bracket size in dimensions and others not.\n\nFor instance, comparing three high-end video cards released in 2020: a Sapphire Radeon RX 5700 XT card measures 135 mm in height (excluding the metal bracket), which exceeds the PCIe standard height by 28 mm, another Radeon RX 5700 XT card by XFX measures 55 mm thick (i.e. 2.7 PCI slots at 20.32 mm), taking up 3 PCIe slots, while an Asus GeForce RTX 3080 video card takes up two slots and measures 318.5mm × 140.1mm × 57.8mm, exceeding PCI Express' maximum length, height, and thickness respectively. The following table identifies the conductors on each side of the edge connector on a PCI Express card. The solder side of the printed circuit board (PCB) is the A-side, and the component side is the B-side. PRSNT1# and PRSNT2# pins must be slightly shorter than the rest, to ensure that a hot-plugged card is fully inserted. The WAKE# pin uses full voltage to wake the computer, but must be pulled high from the standby power to indicate that the card is wake capable. All PCI express cards may consume up to  at  (). The amount of +12 V and total power they may consume depends on the form factor and the role of the card:PCI Express Base Specification, Revision 1.1 Page 332\n* x1 cards are limited to 0.5 A at +12V (6 W) and 10 W combined.\n* x4 and wider cards are limited to 2.1 A at +12V (25 W) and 25 W combined.\n* A full-sized x1 card may draw up to the 25 W limits after initialization and software configuration as a high-power device.\n* A full-sized x16 graphics card may draw up to 5.5 A at +12V (66 W) and 75 W combined after initialization and software configuration as a high-power device.\n\n\n\nOptional connectors add  (6-pin) or  (8-pin) of +12 V power for up to  total ().\n* Sense0 pin is connected to ground by the cable or power supply, or float on board if cable is not connected.\n* Sense1 pin is connected to ground by the cable or power supply, or float on board if cable is not connected.\n\nSome cards use two 8-pin connectors, but this has not been standardized yet , therefore such cards must not carry the official PCI Express logo.  This configuration allows 375 W total () and will likely be standardized by PCI-SIG with the PCI Express 4.0 standard. The 8-pin PCI Express connector could be confused with the EPS12V connector, which is mainly used for powering SMP and multi-core systems. The power connectors are variants of the Molex Mini-Fit Jr. series connectors. Dimensions of PCI Express Mini Cards are 30 mm × 50.95 mm (width × length) for a Full Mini Card. There is a 52-pin edge connector, consisting of two staggered rows on a 0.8 mm pitch.  Each row has eight contacts, a gap equivalent to four contacts, then a further 18 contacts.  Boards have a thickness of 1.0 mm, excluding the components.  A \"Half Mini Card\" (sometimes abbreviated as HMC) is also specified, having approximately half the physical length of 26.8 mm. PCI Express Mini Card edge connectors provide multiple connections and buses:\n* PCI Express x1 (with SMBus)\n* USB 2.0\n* Wires to diagnostics LEDs for wireless network (i.e., Wi-Fi) status on computer's chassis\n* SIM card for GSM and WCDMA applications (UIM signals on spec.)\n* Future extension for another PCIe lane\n* 1.5 V and 3.3 V power M.2 replaces the mSATA standard and Mini PCIe. Computer bus interfaces provided through the M.2 connector are PCI Express 3.0 (up to four lanes), Serial ATA 3.0, and USB 3.0 (a single logical port for each of the latter two). It is up to the manufacturer of the M.2 host or device to choose which interfaces to support, depending on the desired level of host support and device type. OCuLink (standing for \"optical-copper link\", since Cu is the chemical symbol for copper) is an extension for the \"cable version of PCI Express\", designed to compete with Thunderbolt 3. Version 1.0 of OCuLink, released in Oct 2015, supports up to 4 PCIe 3.0 lanes (8 GT/s (gigatransfers per second), 3.9 GB/s) over copper cabling; a fiber optic version may appear in the future.\n\nThe most recent version of OCuLink, OCuLink-2, supports up to 16 GB/s (PCIe 4.0 x8) while the maximum bandwidth of a full speed Thunderbolt 4 cable is 5 GB/s. Some suppliers may design their connector products to support the next-generation PCI Express 5.0, which runs at 32 GT/s per lane, ensuring future proofing and minimizing development costs over the next few years.\n\nWhile initially intended for use in laptops for the connection of powerful external GPU boxes, OCuLink's popularity lies primarily in its use for PCIe interconnections in servers, a more prevalent application. Numerous other form factors use, or are able to use, PCIe. These include:\n* Low-height card\n* ExpressCard: Successor to the PC Card form factor (with x1 PCIe and USB 2.0; hot-pluggable)\n* PCI Express ExpressModule: A hot-pluggable modular form factor defined for servers and workstations\n* XQD card: A PCI Express-based flash card standard by the CompactFlash Association with x2 PCIe\n* CFexpress card: A PCI Express-based flash card by the CompactFlash Association in three form factors supporting 1 to 4 PCIe lanes\n* SD card: The SD Express bus, introduced in version 7.0 of the SD specification uses a x1 PCIe link\n* XMC: Similar to the CMC/PMC form factor (VITA 42.3)\n* AdvancedTCA: A complement to CompactPCI for larger applications; supports serial based backplane topologies\n* AMC: A complement to the AdvancedTCA specification; supports processor and I/O modules on ATCA boards (x1, x2, x4 or x8 PCIe).\n* FeaturePak: A tiny expansion card format (43mm × 65 mm) for embedded and small-form-factor applications, which implements two x1 PCIe links on a high-density connector along with USB, I2C, and up to 100 points of I/O\n* Universal IO: A variant from Super Micro Computer Inc designed for use in low-profile rack-mounted chassis.  It has the connector bracket reversed so it cannot fit in a normal PCI Express socket, but it is pin-compatible and may be inserted if the bracket is removed.\n* M.2 (formerly known as NGFF)\n* M-PCIe brings PCIe 3.0 to mobile devices (such as tablets and smartphones), over the M-PHY physical layer.\n* U.2 (formerly known as SFF-8639)\n\nThe PCIe slot connector can also carry protocols other than PCIe. Some 9xx series Intel chipsets support Serial Digital Video Out, a proprietary technology that uses a slot to transmit video signals from the host CPU's integrated graphics instead of PCIe, using a supported add-in.\n\nThe PCIe transaction-layer protocol can also be used over some other interconnects, which are not electrically PCIe:\n* Thunderbolt: A royalty-free  interconnect standard by Intel that combines DisplayPort and PCIe protocols in a form factor compatible with Mini DisplayPort. Thunderbolt 3.0 also combines USB 3.1 and uses the USB-C form factor as opposed to Mini DisplayPort.\n* USB4 While in early development, PCIe was initially referred to as HSI (for High Speed Interconnect), and underwent a name change to 3GIO (for 3rd Generation I/O) before finally settling on its PCI-SIG name PCI Express. A technical working group named the Arapaho Work Group (AWG) drew up the standard. For initial drafts, the AWG consisted only of Intel engineers; subsequently, the AWG expanded to include industry partners.\n\nSince, PCIe has undergone several large and smaller revisions, improving on performance and other features. ; Notes There are 5 primary releases/checkpoints in a PCI-SIG specification:\n* Draft 0.3 (Concept): this release may have few details, but outlines the general approach and goals.\n* Draft 0.5 (First draft): this release has a complete set of architectural requirements and must fully address the goals set out in the 0.3 draft.\n* Draft 0.7 (Complete draft): this release must have a complete set of functional requirements and methods defined, and no new functionality may be added to the specification after this release. Before the release of this draft, electrical specifications must have been validated via test silicon.\n* Draft 0.9 (Final draft): this release allows PCI-SIG member companies to perform an internal review for intellectual property, and no functional changes are permitted after this draft.\n* 1.0 (Final release): this is the final and definitive specification, and any changes or enhancements are through Errata documentation and Engineering Change Notices (ECNs) respectively.\n\nHistorically, the earliest adopters of a new PCIe specification generally begin designing with the Draft 0.5 as they can confidently build up their application logic around the new bandwidth definition and often even start developing for any new protocol features. At the Draft 0.5 stage, however, there is still a strong likelihood of changes in the actual PCIe protocol layer implementation, so designers responsible for developing these blocks internally may be more hesitant to begin work than those using interface IP from external sources. The PCIe link is built around dedicated unidirectional couples of serial (1-bit), point-to-point connections known as lanes. This is in sharp contrast to the earlier PCI connection, which is a bus-based system where all the devices share the same bidirectional, 32-bit or 64-bit parallel bus.\n\nPCI Express is a layered protocol, consisting of a transaction layer, a data link layer, and a physical layer. The Data Link Layer is subdivided to include a media access control (MAC) sublayer.  The Physical Layer is subdivided into logical and electrical sublayers. The Physical logical-sublayer contains a physical coding sublayer (PCS). The terms are borrowed from the IEEE 802 networking protocol model. PCIe sends all control messages, including interrupts, over the same links used for data. The serial protocol can never be blocked, so latency is still comparable to conventional PCI, which has dedicated interrupt lines. When the problem of IRQ sharing of pin based interrupts is taken into account and the fact that message signaled interrupts (MSI) can bypass an I/O APIC and be delivered to the CPU directly, MSI performance ends up being substantially better.\n\nData transmitted on multiple-lane links is interleaved, meaning that each successive byte is sent down successive lanes. The PCIe specification refers to this interleaving as data striping. While requiring significant hardware complexity to synchronize (or deskew) the incoming striped data, striping can significantly reduce the latency of the nth byte on a link. While the lanes are not tightly synchronized, there is a limit to the lane to lane skew of 20/8/6 ns for 2.5/5/8 GT/s so the hardware buffers can re-align the striped data. Due to padding requirements, striping may not necessarily reduce the latency of small data packets on a link.\n\nAs with other high data rate serial transmission protocols, the clock is embedded in the signal. At the physical level, PCI Express 2.0 utilizes the 8b/10b encoding scheme (line code) to ensure that strings of consecutive identical digits (zeros or ones) are limited in length. This coding was used to prevent the receiver from losing track of where the bit edges are. In this coding scheme every eight (uncoded) payload bits of data are replaced with 10 (encoded) bits of transmit data, causing a 20% overhead in the electrical bandwidth. To improve the available bandwidth, PCI Express version 3.0 instead uses 128b/130b encoding (1.54% overhead). Line encoding limits the run length of identical-digit strings in data streams and ensures the receiver stays synchronised to the transmitter via clock recovery.\n\nA desirable balance (and therefore spectral density) of 0 and 1 bits in the data stream is achieved by XORing a known binary polynomial as a \"scrambler\" to the data stream in a feedback topology. Because the scrambling polynomial is known, the data can be recovered by applying the XOR a second time. Both the scrambling and descrambling steps are carried out in hardware. The data link layer performs three vital services for the PCIe link:\n# sequence the transaction layer packets (TLPs) that are generated by the transaction layer,\n# ensure reliable delivery of TLPs between two endpoints via an acknowledgement protocol (ACK and NAK signaling) that explicitly requires replay of unacknowledged/bad TLPs,\n# initialize and manage flow control credits\n\nOn the transmit side, the data link layer generates an incrementing sequence number for each outgoing TLP. It serves as a unique identification tag for each transmitted TLP, and is inserted into the header of the outgoing TLP.  A 32-bit cyclic redundancy check code (known in this context as Link CRC or LCRC) is also appended to the end of each outgoing TLP.\n\nOn the receive side, the received TLP's LCRC and sequence number are both validated in the link layer. If either the LCRC check fails (indicating a data error), or the sequence-number is out of range (non-consecutive from the last valid received TLP), then the bad TLP, as well as any TLPs received after the bad TLP, are considered invalid and discarded.  The receiver sends a negative acknowledgement message (NAK) with the sequence-number of the invalid TLP, requesting re-transmission of all TLPs forward of that sequence-number. If the received TLP passes the LCRC check and has the correct sequence number, it is treated as valid. The link receiver increments the sequence-number (which tracks the last received good TLP), and forwards the valid TLP to the receiver's transaction layer.  An ACK message is sent to remote transmitter, indicating the TLP was successfully received (and by extension, all TLPs with past sequence-numbers.)\n\nIf the transmitter receives a NAK message, or no acknowledgement (NAK or ACK) is received until a timeout period expires, the transmitter must retransmit all TLPs that lack a positive acknowledgement (ACK).  Barring a persistent malfunction of the device or transmission medium, the link-layer presents a reliable connection to the transaction layer, since the transmission protocol ensures delivery of TLPs over an unreliable medium.\n\nIn addition to sending and receiving TLPs generated by the transaction layer, the data-link layer also generates and consumes data link layer packets (DLLPs). ACK and NAK signals are communicated via DLLPs, as are some power management messages and flow control credit information (on behalf of the transaction layer).\n\nIn practice, the number of in-flight, unacknowledged TLPs on the link is limited by two factors: the size of the transmitter's replay buffer (which must store a copy of all transmitted TLPs until the remote receiver ACKs them), and the flow control credits issued by the receiver to a transmitter. PCI Express requires all receivers to issue a minimum number of credits, to guarantee a link allows sending PCIConfig TLPs and message TLPs. PCI Express implements split transactions (transactions with request and response separated by time), allowing the link to carry other traffic while the target device gathers data for the response.\n\nPCI Express uses credit-based flow control. In this scheme, a device advertises an initial amount of credit for each received buffer in its transaction layer. The device at the\nopposite end of the link, when sending transactions to this device, counts the number of credits each TLP consumes from its account. The sending device may only transmit a TLP when doing so does not make its consumed credit count exceed its credit limit. When the receiving device finishes processing the TLP from its buffer, it signals a return of credits to the sending device, which increases the credit limit by the restored amount.  The credit counters are modular counters, and the comparison of consumed credits to credit limit requires modular arithmetic. The advantage of this scheme (compared to other methods such as wait states or handshake-based transfer protocols) is that the latency of credit return does not affect performance, provided that the credit limit is not encountered. This assumption is generally met if each device is designed with adequate buffer sizes.\n\nPCIe 1.x is often quoted to support a data rate of 250 MB/s in each direction, per lane. This figure is a calculation from the physical signaling rate (2.5 gigabaud) divided by the encoding overhead (10 bits per byte). This means a sixteen lane (x16) PCIe card would then be theoretically capable of 16x250 MB/s = 4 GB/s in each direction. While this is correct in terms of data bytes, more meaningful calculations are based on the usable data payload rate, which depends on the profile of the traffic, which is a function of the high-level (software) application and intermediate protocol levels.\n\nLike other high data rate serial interconnect systems, PCIe has a protocol and processing overhead due to the additional transfer robustness (CRC and acknowledgements).  Long continuous unidirectional transfers (such as those typical in high-performance storage controllers) can approach >95% of PCIe's raw (lane) data rate. These transfers also benefit the most from increased number of lanes (x2, x4, etc.) But in more typical applications (such as a USB or Ethernet controller), the traffic profile is characterized as short data packets with frequent enforced acknowledgements. This type of traffic reduces the efficiency of the link, due to overhead from packet parsing and forced interrupts (either in the device's host interface or the PC's CPU).  Being a protocol for devices connected to the same printed circuit board, it does not require the same tolerance for transmission errors as a protocol for communication over longer distances, and thus, this loss of efficiency is not particular to PCIe. As for any \"network like\" communication links, some of the \"raw\" bandwidth is consumed by protocol overhead:\n\nA PCIe 1.x lane for example offers a data rate on top of the physical layer of 250 MB/s (simplex). This isn't the payload bandwidth but the physical layer bandwidth – a PCIe lane has to carry additional information for full functionality.\n\n\nThe Gen2 overhead is then 20, 24, or 28 bytes per transaction.\n\n\nThe Gen3 overhead is then 22, 26 or 30 bytes per transaction.\n\nThe \\text{Packet Efficiency} = \\frac{\\text{Payload}}{\\text{Payload} + \\text{Overhead}} for a 128 byte payload is 86%, and 98% for a 1024 byte payload. For small accesses like register settings (4 bytes), the efficiency drops as low as 16%.\n\nThe maximum payload size (MPS) is set on all devices based on smallest maximum on any device in the chain.  If one device has an MPS of 128 bytes, all devices of the tree must set their MPS to 128 bytes.  In this case the bus will have a peak efficiency of 86% for writes. PCI Express operates in consumer, server, and industrial applications, as a motherboard-level interconnect (to link motherboard-mounted peripherals), a passive backplane interconnect and as an expansion card interface for add-in boards.\n\nIn virtually all modern () PCs, from consumer laptops and desktops to enterprise data servers, the PCIe bus serves as the primary motherboard-level interconnect, connecting the host system-processor with both integrated peripherals (surface-mounted ICs) and add-on peripherals (expansion cards). In most of these systems, the PCIe bus co-exists with one or more legacy PCI buses, for backward compatibility with the large body of legacy PCI peripherals.\n\n, PCI Express has replaced AGP as the default interface for graphics cards on new systems. Almost all models of graphics cards released since 2010 by AMD (ATI) and Nvidia use PCI Express. Nvidia uses the high-bandwidth data transfer of PCIe for its Scalable Link Interface (SLI) technology, which allows multiple graphics cards of the same chipset and model number to run in tandem, allowing increased performance.  AMD has also developed a multi-GPU system based on PCIe called CrossFire. AMD, Nvidia, and Intel have released motherboard chipsets that support as many as four PCIe x16 slots, allowing tri-GPU and quad-GPU card configurations. Theoretically, external PCIe could give a notebook the graphics power of a desktop, by connecting a notebook with any PCIe desktop video card (enclosed in its own external housing, with a power supply and cooling); this is possible with an ExpressCard or Thunderbolt interface. An ExpressCard interface provides bit rates of 5 Gbit/s (0.5 GB/s throughput), whereas a Thunderbolt interface provides bit rates of up to 40 Gbit/s (5 GB/s throughput).\n\nIn 2006, Nvidia developed the Quadro Plex external PCIe family of GPUs that can be used for advanced graphic applications for the professional market. These video cards require a PCI Express x8 or x16 slot for the host-side card, which connects to the Plex via a VHDCI carrying eight PCIe lanes.\n\nIn 2008, AMD announced the ATI XGP technology, based on a proprietary cabling system that is compatible with PCIe x8 signal transmissions. This connector is available on the Fujitsu Amilo and the Acer Ferrari One notebooks. Fujitsu launched their AMILO GraphicBooster enclosure for XGP soon thereafter. Around 2010 Acer launched the Dynavivid graphics dock for XGP.\n\nIn 2010, external card hubs were introduced that can connect to a laptop or desktop through a PCI ExpressCard slot. These hubs can accept full-sized graphics cards.  Examples include MSI GUS, Village Instrument's ViDock, the Asus XG Station, Bplus PE4H V3.2 adapter, as well as more improvised DIY devices. However such solutions are limited by the size (often only x1) and version of the available PCIe slot on a laptop.\n\nThe Intel Thunderbolt interface has provided a new option to connect with a PCIe card externally. Magma has released the ExpressBox 3T, which can hold up to three PCIe cards (two at x8 and one at x4).  MSI also released the Thunderbolt GUS II, a PCIe chassis dedicated for video cards. Other products such as the Sonnet's Echo Express and mLogic's mLink are Thunderbolt PCIe chassis in a smaller form factor.\n\nIn 2017, more fully featured external card hubs were introduced, such as the Razer Core, which has a full-length PCIe x16 interface. The PCI Express protocol can be used as data interface to flash memory devices, such as memory cards and solid-state drives (SSDs).\n\nThe XQD card is a memory card format utilizing PCI Express, developed by the CompactFlash Association, with transfer rates of up to 1GB/s.\n\nMany high-performance, enterprise-class SSDs are designed as PCI Express RAID controller cards.  Before NVMe was standardized, many of these cards utilized proprietary interfaces and custom drivers to communicate with the operating system; they had much higher transfer rates (over 1 GB/s) and IOPS (over one million I/O operations per second) when compared to Serial ATA or SAS drives. For example, in 2011 OCZ and Marvell co-developed a native PCI Express solid-state drive controller for a PCI Express 3.0 x16 slot with maximum capacity of 12 TB and a performance of to 7.2 GB/s sequential transfers and up to 2.52 million IOPS in random transfers.\n\nSATA Express was an interface for connecting SSDs through SATA-compatible ports, optionally providing multiple PCI Express lanes as a pure PCI Express connection to the attached storage device.  M.2 is a specification for internally mounted computer expansion cards and associated connectors, which also uses multiple PCI Express lanes.\n\nPCI Express storage devices can implement both AHCI logical interface for backward compatibility, and NVM Express logical interface for much faster I/O operations provided by utilizing internal parallelism offered by such devices.  Enterprise-class SSDs can also implement SCSI over PCI Express. Certain data-center applications (such as large computer clusters) require the use of fiber-optic interconnects due to the distance limitations inherent in copper cabling. Typically, a network-oriented standard such as Ethernet or Fibre Channel suffices for these applications, but in some cases the overhead introduced by routable protocols is undesirable and a lower-level interconnect, such as InfiniBand, RapidIO, or NUMAlink is needed.  Local-bus standards such as PCIe and HyperTransport can in principle be used for this purpose, but , solutions are only available from niche vendors such as Dolphin ICS, and TTTech Auto. Other communications standards based on high bandwidth serial architectures include InfiniBand, RapidIO, HyperTransport, Intel QuickPath Interconnect, and the Mobile Industry Processor Interface (MIPI). The differences are based on the trade-offs between flexibility and extensibility vs latency and overhead.  For example, making the system hot-pluggable, as with Infiniband but not PCI Express, requires that software track network topology changes.\n\nAnother example is making the packets shorter to decrease latency (as is required if a bus must operate as a memory interface). Smaller packets mean packet headers consume a higher percentage of the packet, thus decreasing the effective bandwidth. Examples of bus protocols designed for this purpose are RapidIO and HyperTransport.\n\nPCI Express falls somewhere in the middle, targeted by design as a system interconnect (local bus) rather than a device interconnect or routed network protocol. Additionally, its design goal of software transparency constrains the protocol and raises its latency somewhat.\n\nDelays in PCIe 4.0 implementations led to the Gen-Z consortium, the CCIX effort and an open Coherent Accelerator Processor Interface (CAPI) all being announced by the end of 2016.\n\nOn 11 March 2019, Intel presented Compute Express Link (CXL), a new interconnect bus, based on the PCI Express 5.0 physical layer infrastructure. The initial promoters of the CXL specification included: Alibaba, Cisco, Dell EMC, Facebook, Google, HPE, Huawei, Intel and Microsoft. The PCI-SIG Integrators List lists products made by PCI-SIG member companies that have passed compliance testing. The list include switches, bridges, NICs, SSDs, etc. * Active State Power Management (ASPM)\n* Peripheral Component Interconnect\n* PCI configuration space\n* PCI-X\n* PCI/104-Express\n* PCIe/104\n* Root complex\n* Serial Digital Video Out (SDVO)\n* \n* UCIe\n* Compute Express Link (CXL) * , 1120 pp.\n* , 1056 pp.\n* , 325 pp. * \n* PCI-SIG Specifications\n\n\n\n\n\nCategory:Computer-related introductions in 2004\nCategory:Peripheral Component Interconnect\nCategory:Serial buses\nCategory:Computer standards\nCategory:Motherboard expansion slot",
)

example_table_3 = {
        "PCI card type": ["Full-Length", "Half-Length", "Low-Profile/Slim"],
        "Dimensions height × length × width, maximum (mm)": ["111.15 × 312.00 × 20.32", "111.15 × 167.65 × 20.32", "68.90 × 167.65 × 20.32"],
        "Dimensions height × length × width, maximum (in)": ["4.376 × 12.283 × 0.8", "4.376 × 6.600 × 0.8", "2.731 × 6.600 × 0.8"]
      }

example_mapping_3 = {
      "Dimensions height × length × width, maximum (in)": "P9767",
      "Dimensions height × length × width, maximum (mm)": "P2048",
      "PCI card type": "P2049"
    }


def clean_title(title):
    if not title:
        return ""
    title = str(title).strip()
    title = re.sub(r"\s*\d+$", "", title)
    return title.strip()


def clean_text(text):
    if not text:
        return ""

    text = str(text).strip()
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d{4}(-\d{4})?\)", "", text)
    text = text.replace("–", "-").strip()

    return text


def safe_json_extract(raw_text):
    """
    Extract JSON from model output.
    Keeps your logic unchanged.
    """
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)

    if match:
        raw_text = match.group(1)

    try:
        obj = json.loads(raw_text)
    except:
        obj = {}

    for key in ["Predicate mapping", "Predicate_mapping"]:
        if key in obj:
            obj = obj[key]

    obj.pop("Explanation", None)

    return obj


# ============================================================
# STAGE 1: COLUMN -> PREDICATE MAPPING
# ============================================================

def get_mapping(new_table, new_text, new_title):

    prompt_text = f"""
        You are an AI assistant that maps table columns to knowledge graph predicates.

        Allowed predicates:
        {chr(10).join([f"- {k}: {v}" for k, v in allowed_predicates.items()])}

        --------------------------------------------------
        Example 1 (Surface Studio)
        Table columns:
        {chr(10).join([f"- {k}: {v}" for k, v in example_table_1.items()])}
        Text:
        {example_text_1}
        Relevant entity: Surface Studio (may appear as subject or object)
        Explanation:
        - "Surface Studio Configuration Options Microsoft Surface Studio tech specs Price Tier (USD)":
          - Table: Lists $2,999, $3,499, $4,199 for different configurations.
          - Text: Mentions “starting at $2,999” and compares different tiers.
          - Reasoning → These values represent the cost of the product → mapped to price (P2284).
        - "Surface Studio Configuration Options Microsoft Surface Studio tech specs CPU":
          - Table: Lists Intel Core i5-6440HQ, i7-6820HQ processors for each configuration.
          - Text: States “6th generation (codename ‘Skylake’) Intel Core i5 or Core i7 processor.”
          - Reasoning → These are the central processing units used in the product → mapped to CPU (P880).
        - "Surface Studio Configuration Options Microsoft Surface Studio tech specs Integrated GPU":
          - Table: Lists GTX 965M and GTX 980M graphics processors.
          - Text: Mentions “either a NVIDIA GeForce GTX 965M or GeForce GTX 980M graphics processor.”
          - Reasoning → These are the graphics processing units within the system → mapped to GPU (P2560).
        - "Surface Studio Configuration Options Microsoft Surface Studio tech specs RAM":
          - Table: Lists 8 GB, 16 GB, 32 GB.
          - Text: Mentions “up to 32 GB of DDR4 RAM.”
          - Reasoning → These values indicate the device’s RAM capacity → mapped to RAM capacity (P13525).
        - "Surface Studio Configuration Options Microsoft Surface Studio tech specs Internal Storage":
          - Table: Lists combinations like 1 TB SATA II HDD + 64 GB SSD, 2 TB HDD + 128 GB SSD.
          - Text: Mentions “2 terabyte hard drive” and storage configurations.
          - Reasoning → These values indicate storage capacity of the device → mapped to storage capacity (P2928).
        Predicate mapping:
        {json.dumps(example_mapping_1, indent=2)}

        --------------------------------------------------
        Example 2 (WaveLAN)
        Table columns:
        {chr(10).join([f"- {k}: {v}" for k, v in example_table_2.items()])}
        Text:
        {example_text_2}
        Relevant entity: WaveLAN (may appear as subject or object)
        Explanation:
        - "WaveLAN Technical Specifications – Frequency":
          - Table: Lists frequencies for each region, e.g., 915 MHz for US & Canada, 2.412–2.462 GHz for Worldwide, 2.422–2.462 GHz for Europe, 2.460–2.462 GHz for France, 2.422–2.442 GHz for Australia, 2.484 GHz for Japan.
          - Text: Describes Classic WaveLAN operating in 900 MHz or 2.4 GHz ISM bands, with later 2.4 GHz cards capable of multiple channels (2.412–2.484 GHz) depending on region-specific firmware.
          - Reasoning → These values represent the operating radio frequencies of WaveLAN devices in different regions → mapped to Frequency (P2144).
        - "WaveLAN Technical Specifications – Security":
          - Table: Shows “16-bit network ID and optional DES encryption” for all regions.
          - Text: Explains WaveLAN security: 16-bit NWID, optional DES encryption on ISA/MCA cards and WavePoint access points, and WEP for IEEE 802.11 cards; notes the later transition to WPA.
          - Reasoning → These values indicate the methods used to secure WaveLAN communications → mapped to Security (P2283).
        - "WaveLAN Technical Specifications – Output power":
          - Table: Lists output power by region: 250 mW for US & Canada (900 MHz), 32 mW for all 2.4 GHz regions.
          - Text: Confirms these transmission power values for Classic and IEEE 802.11 WaveLAN cards in their respective frequency bands.
          - Reasoning → These numbers describe the transmission strength of the wireless devices → mapped to Output power (P2109).
        - "WaveLAN Technical Specifications – Maximum data rate":
          - Table: All regions list 2 Mbit/s.
          - Text: Mentions Classic WaveLAN cards and early IEEE 802.11 cards operate at 2 Mbit/s.
          - Reasoning → These values represent the maximum data throughput for WaveLAN devices → mapped to Maximum data rate (P6711).
        Predicate mapping:
        {json.dumps(example_mapping_2, indent=2)}

        --------------------------------------------------
        Example 3 (PCI Express)
        Table columns:
        {chr(10).join([f"- {k}: {v}" for k, v in example_table_3.items()])}
        Text:
        {example_text_3}
        Relevant entity: PCI Express (may appear as subject or object)
        Explanation:
        - "PCI card type":
          - Table: Lists types of PCIe cards: Full-Length, Half-Length, Low-Profile/Slim.
          - Text: Explains that PCIe expansion cards come in different physical form factors to fit various motherboard slots and case sizes; Full-Length cards are typically used for high-performance graphics or RAID controllers, Half-Length for standard cards, and Low-Profile/Slim for compact systems.
          - Reasoning → These values define the physical form factor of PCI Express cards → mapped to PCI card type (P2049).
        - "Dimensions height × length × width, maximum (mm)":
          - Table: Lists maximum dimensions for each card type in millimeters: Full-Length 111.15 × 312.00 × 20.32 mm, Half-Length 111.15 × 167.65 × 20.32 mm, Low-Profile/Slim 68.90 × 167.65 × 20.32 mm.
          - Text: Confirms the maximum allowable physical dimensions for PCIe cards according to the PCI Express specification, accommodating standard slot sizes and ensuring compatibility with typical desktop and server cases.
          - Reasoning → These measurements specify the physical size limits for PCI Express cards → mapped to Dimensions height × length × width, maximum (mm) (P2048).
        - "Dimensions height × length × width, maximum (in)":
          - Table: Lists maximum dimensions in inches: Full-Length 4.376 × 12.283 × 0.8 in, Half-Length 4.376 × 6.600 × 0.8 in, Low-Profile/Slim 2.731 × 6.600 × 0.8 in.
          - Text: Provides equivalent maximum dimensions in inches for international reference, ensuring correct physical design and manufacturing standards.
          - Reasoning → These values are the imperial equivalents of the card size limits → mapped to Dimensions height × length × width, maximum (in) (P9767).
        Predicate mapping:
        {json.dumps(example_mapping_3, indent=2)}

        --------------------------------------------------
        New Table - {new_title}
        Table columns:
        {chr(10).join([f"- {k}: {v}" for k, v in new_table.items()])}
        Text:
        {new_text or "No additional text provided."}
        Relevant entity: {new_title} (may appear as subject or object)

        Instructions for mapping:
        - Map each table column to one of the allowed predicates.
        - Use table values as the primary clue. Use text only to clarify ambiguous columns.
        - Columns with no clear predicate → None.
        - Return mapping in json format. No extra text or explanation.

        Additional Guidance:
        1. Monetary / Price Columns:
          - Columns like "Price", "Cost", "Launch Price", "Tier" → map to price-related predicates.
          - Currency or unit clues in the table help confirm.

        2. Date Columns:
          - Columns like "Released", "Launched", "Discontinued", "Updated" → map to release or end date predicates.
          - Past tense in text helps distinguish discontinued items.

        3. Hardware / Technical Specs:
          - Columns like "CPU", "RAM", "Storage", "GPU", "Processor Speed" → map to technical predicates.
          - Units like GB, MHz, or TB can help confirm.

        4. Person / Organization Names:
          - Columns like "CEO", "Chairperson", "Founder" → map to person-related predicates.
          - Ownership/affiliation phrases in text: past tense → acquired, present tense → subsidiary.

        5. Frequency / Power / Performance:
          - Columns like "Frequency", "Data rate", "Output power" → map to performance or signal predicates.

        6. Attribute-only Columns:
          - Columns like "Notes", "Description", "Industry", "Website", "Key people", "Products" → None.

        7. Use Text Only for Disambiguation:
          - Table column names are primary. Text is secondary to clarify meaning when ambiguous.

        8. Default When Unsure:
          - If column meaning is unclear and text is missing, map to None.
        """

    # Send request to OpenAI ChatCompletion API
    llm_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=512,
        temperature=0,
    )
    raw_output = llm_answer.choices[0].message.content

    return safe_json_extract(raw_output)


# ============================================================
# FULL PIPELINE
# ============================================================

def run_system(input_file):

    print(f"\nLoading input file: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    rows = []

    # --------------------------------------------------------
    # LOOP THROUGH INPUT DATA
    # --------------------------------------------------------
    for doc_entry in tqdm(json_data):

        doc_id = doc_entry.get("id")

        table_obj = doc_entry.get("table", {})
        table_data = table_obj.get("tableData", {})

        if not table_data:
            continue

        title = doc_entry.get("doc_title", "")
        title_clean = clean_title(title)

        if not title_clean:
            title_clean = doc_id

        text = doc_entry.get("full_text", "")

        print(f"\nProcessing table from doc_id: {doc_id}")

        # ====================================================
        # STAGE 1: PREDICATE MAPPING
        # ====================================================
        try:
            mapping = get_mapping(table_data, text, title_clean)
        except Exception as e:
            print("Mapping error:", e)
            continue

        # Fixes preserved from your cleaning script
        cleaned_mapping = {}

        for col, predicate in mapping.items():

            cleaned_mapping[col] = predicate

        # ====================================================
        # STAGE 2: TRIPLE GENERATION
        # ====================================================
        for column, values in table_data.items():

            predicate = cleaned_mapping.get(column)

            if not predicate or predicate == "None":
                continue

            if not isinstance(values, list):
                values = [values]

            for value in values:

                if value is None or str(value).strip() == "":
                    continue

                val = clean_text(value)

                if not val:
                    continue

                subject = title_clean
                obj = val

                rows.append({
                    "document_id": doc_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })

        time.sleep(2)

        df = pd.DataFrame(rows)
        if not df.empty:
            df.drop_duplicates(
                subset=["document_id", "subject", "predicate", "object"],
                inplace=True
            )
            print("\n================ FINAL RDF TRIPLES ================\n")
            print(df.to_string(index=False))

            print(f"\nTotal triples generated: {len(df)}")
        else:
            print("\n No triples generated for this table.")


if __name__ == "__main__":

    input_path = "cleaned_data/telco_data.json"

    run_system(input_path)