// Portions of this file are derived from ecgfp5
// Copyright (c) 2022 Thomas Pornin
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

// Portions of this file are derived from plonky2-ecgfp5
// Copyright (c) 2023 Sebastien La Duca
// Licensed under the MIT License. See THIRD_PARTY_NOTICES for details.

use plonky2::field::extension::quintic::QuinticExtension;

use super::curve::AffinePoint;
use crate::types::config::const_f;

pub(crate) const MUL_TABLE_G0: [AffinePoint; 16] = [
    AffinePoint {
        x: QuinticExtension([
            const_f(12883135586176881569),
            const_f(4356519642755055268),
            const_f(5248930565894896907),
            const_f(2165973894480315022),
            const_f(2448410071095648785),
        ]),
        u: QuinticExtension([
            const_f(13835058052060938241),
            const_f(0),
            const_f(0),
            const_f(0),
            const_f(0),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(16517537419581740386),
            const_f(6962630169123120981),
            const_f(12147752690379666704),
            const_f(16637325971742264607),
            const_f(2335078582315237010),
        ]),
        u: QuinticExtension([
            const_f(8457587110646932172),
            const_f(138591869800252458),
            const_f(3187444967472352324),
            const_f(18179149801168653736),
            const_f(9453003655195557048),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(4546139357324501584),
            const_f(1393728687664685160),
            const_f(15208040286522119521),
            const_f(7903224051455420834),
            const_f(12463930627278381774),
        ]),
        u: QuinticExtension([
            const_f(16373828487211693378),
            const_f(5899455736915524900),
            const_f(17616512450102495476),
            const_f(17643201028570366669),
            const_f(2833280130550676525),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(4341836049185169731),
            const_f(9111482874850194930),
            const_f(7798994609726992878),
            const_f(12619124383509403661),
            const_f(13047834166950680886),
        ]),
        u: QuinticExtension([
            const_f(3584786391427904733),
            const_f(1717626083626375072),
            const_f(16549008311909030594),
            const_f(17550175197111849143),
            const_f(18374971670674568416),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(18121072711119258927),
            const_f(3394315639035318724),
            const_f(2648370499809919556),
            const_f(13348924736921714137),
            const_f(3428166646246873447),
        ]),
        u: QuinticExtension([
            const_f(9264305576790077869),
            const_f(7426254234280836405),
            const_f(5107777768036114824),
            const_f(9390769538758625122),
            const_f(9788182195111344062),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(11080635543643017332),
            const_f(3122290570793204485),
            const_f(16632474826839786439),
            const_f(14883711538614796285),
            const_f(10396852362099782295),
        ]),
        u: QuinticExtension([
            const_f(14253916706639980511),
            const_f(15728038457561632290),
            const_f(3947138785484546318),
            const_f(4740958322851071718),
            const_f(17384736114265519442),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(4763058716218401568),
            const_f(17879823368956058516),
            const_f(13578954599286698938),
            const_f(8634670560943921567),
            const_f(13706660844700767685),
        ]),
        u: QuinticExtension([
            const_f(3354778288360932917),
            const_f(13842278303693121409),
            const_f(4717821645259836467),
            const_f(7978743897613094276),
            const_f(10118963888992569394),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(4026958896735257282),
            const_f(13595990041314210204),
            const_f(11499471878438064392),
            const_f(10019455879458851233),
            const_f(11986847968355927330),
        ]),
        u: QuinticExtension([
            const_f(14532821659997761913),
            const_f(9582789969382797985),
            const_f(3082219099923033594),
            const_f(2859656980617778370),
            const_f(3746047816071136016),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(15935900828168308224),
            const_f(8668680449802005535),
            const_f(491315506768012688),
            const_f(6584881037682113026),
            const_f(12386385009372860460),
        ]),
        u: QuinticExtension([
            const_f(13217832923050551864),
            const_f(51671271962049328),
            const_f(15400792709153778477),
            const_f(6752203529649104660),
            const_f(2855313280735340066),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(8473506523195244465),
            const_f(2446964921175324878),
            const_f(17962771942831363202),
            const_f(6949608686158330138),
            const_f(9315492999547366751),
        ]),
        u: QuinticExtension([
            const_f(5171814696081600409),
            const_f(3025466154945175207),
            const_f(453302446979841822),
            const_f(14135305892339872079),
            const_f(2556388051049291052),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(3960231187580500028),
            const_f(3695840168764199059),
            const_f(2914577777792670911),
            const_f(9249939676680902688),
            const_f(17553522813502241416),
        ]),
        u: QuinticExtension([
            const_f(3015152305907361949),
            const_f(10730034543155667220),
            const_f(3314242046485170944),
            const_f(1984395553885795852),
            const_f(13781645774758249860),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(11575997426281090678),
            const_f(1534495174840625570),
            const_f(7539338128385981583),
            const_f(10393042019577161985),
            const_f(10667466219175771157),
        ]),
        u: QuinticExtension([
            const_f(16681365912970185037),
            const_f(11287896019745355117),
            const_f(11069899752345274504),
            const_f(15487604769605237513),
            const_f(13467978440572613228),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(11192179397773394280),
            const_f(3555953455665397909),
            const_f(5346523552109387121),
            const_f(4514445299325204396),
            const_f(3932728981135688453),
        ]),
        u: QuinticExtension([
            const_f(5421638117266109845),
            const_f(204299445119713184),
            const_f(6067390115784997081),
            const_f(16191134954342419157),
            const_f(4139938600224417293),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(13189785832536261642),
            const_f(8777097377506996162),
            const_f(17497140949916325738),
            const_f(15140279769427597032),
            const_f(15517274717131999881),
        ]),
        u: QuinticExtension([
            const_f(1040464435413162742),
            const_f(9262701069034606854),
            const_f(2990438819650713743),
            const_f(18129195737333990255),
            const_f(12490074042478236606),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(17716508479149156535),
            const_f(14351380558651795729),
            const_f(3644546258883003807),
            const_f(5171318241596472386),
            const_f(294806796132518330),
        ]),
        u: QuinticExtension([
            const_f(7535225611936271281),
            const_f(14682077054502188499),
            const_f(784215514926156349),
            const_f(5280586574139275596),
            const_f(14407528916988559545),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(8681294642569802563),
            const_f(7751765660802747503),
            const_f(16382129702876313971),
            const_f(7447155060842833278),
            const_f(6859908403876474879),
        ]),
        u: QuinticExtension([
            const_f(9674486254207846385),
            const_f(5248970165164951259),
            const_f(3611784478790504991),
            const_f(18437168019170350173),
            const_f(3537959913875671086),
        ]),
    },
];
pub(crate) const MUL_TABLE_G40: [AffinePoint; 16] = [
    AffinePoint {
        x: QuinticExtension([
            const_f(6996996617034310847),
            const_f(1312534891996392328),
            const_f(1967056454231743182),
            const_f(12432745115107639465),
            const_f(8188918658769983203),
        ]),
        u: QuinticExtension([
            const_f(9779151955752388390),
            const_f(12827693252247248589),
            const_f(8299002358494291091),
            const_f(10057624387258292793),
            const_f(9561932552523598817),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(12727761422252171591),
            const_f(11715233354058649362),
            const_f(11258110171296383015),
            const_f(4946612044061620143),
            const_f(10674140266605092092),
        ]),
        u: QuinticExtension([
            const_f(13968556698015688219),
            const_f(9764817221409883159),
            const_f(6009815048702102249),
            const_f(928542484379469501),
            const_f(17548136021451934003),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(1449347403861973234),
            const_f(1268700206465777189),
            const_f(68931803832001930),
            const_f(508124187869777281),
            const_f(14966299269768645002),
        ]),
        u: QuinticExtension([
            const_f(12519156548432608657),
            const_f(1830718924858545317),
            const_f(8290101973558828816),
            const_f(6963396969528752135),
            const_f(5027294278125306748),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(7543060992125635458),
            const_f(13154269169488238929),
            const_f(13038629689209617296),
            const_f(17607665244721587028),
            const_f(4076634695833139117),
        ]),
        u: QuinticExtension([
            const_f(16474278336963843968),
            const_f(7342735040871703005),
            const_f(11822823161099820577),
            const_f(15838689010341349421),
            const_f(8387592947884092077),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6956288471673670775),
            const_f(10363521468804730232),
            const_f(1618310362752793530),
            const_f(16886810269476841179),
            const_f(4982980062158920723),
        ]),
        u: QuinticExtension([
            const_f(13688045661223437644),
            const_f(17947601766473933193),
            const_f(7138906029562123225),
            const_f(14564553876341839060),
            const_f(4126496432434298977),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(800685292854622487),
            const_f(11040079590365906652),
            const_f(1466305609865524328),
            const_f(8372552820474238249),
            const_f(10874913568038030998),
        ]),
        u: QuinticExtension([
            const_f(4703724548613471267),
            const_f(16058989380922585526),
            const_f(8365972383552432650),
            const_f(12321780682158893877),
            const_f(2418487585371688136),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(12518522439291713316),
            const_f(18265859802652833974),
            const_f(4355197864757715454),
            const_f(16333106890933317228),
            const_f(7860825917869078801),
        ]),
        u: QuinticExtension([
            const_f(12169474924601364130),
            const_f(1427729574788767322),
            const_f(3451823787886833090),
            const_f(4595725973834664846),
            const_f(5636506224235047729),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(7016248550036618856),
            const_f(14664150699534918598),
            const_f(7289196844394571239),
            const_f(3733481542224777638),
            const_f(12940819275544993154),
        ]),
        u: QuinticExtension([
            const_f(5962170105887193190),
            const_f(7757792046810148121),
            const_f(17754145760690637154),
            const_f(5608151523576337415),
            const_f(10158975094989974837),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(9657667902075638078),
            const_f(1738398137082324930),
            const_f(3309896085425426006),
            const_f(5244596195331513559),
            const_f(11098614916240915598),
        ]),
        u: QuinticExtension([
            const_f(10176686769986870501),
            const_f(17149616066773579692),
            const_f(16557806655360885458),
            const_f(6409371822017281510),
            const_f(447032970886916415),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(2000298634105946483),
            const_f(7990437998469847294),
            const_f(13891384442822604159),
            const_f(3400617664053350732),
            const_f(17650120710895099722),
        ]),
        u: QuinticExtension([
            const_f(9011065287270146338),
            const_f(9712006535696787670),
            const_f(5197636265344816024),
            const_f(14644619822912127741),
            const_f(5091497898426581809),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(4353689210628214181),
            const_f(11629282537514442736),
            const_f(519301038092536110),
            const_f(17451856528277649540),
            const_f(8053963837814854762),
        ]),
        u: QuinticExtension([
            const_f(16247175863676166340),
            const_f(13321024650071188595),
            const_f(13226465566647040787),
            const_f(15830701216342305199),
            const_f(10171768200911815007),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(8618084654605828418),
            const_f(2932134432657893516),
            const_f(14040921219429416616),
            const_f(8539270659386774297),
            const_f(8223174716536738537),
        ]),
        u: QuinticExtension([
            const_f(2118173438466787625),
            const_f(17017456632539625481),
            const_f(3822614388660837302),
            const_f(18012676134277779138),
            const_f(14555233257002087745),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(14104364668871773315),
            const_f(4671583541374529339),
            const_f(14595315310536253921),
            const_f(12293043219805252275),
            const_f(11083273927620890457),
        ]),
        u: QuinticExtension([
            const_f(13013197605833180311),
            const_f(6369553806055216484),
            const_f(13715364943719691230),
            const_f(832870131890809214),
            const_f(2834204446065110889),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(17610707880266706457),
            const_f(8946198449628536102),
            const_f(14056449117236625467),
            const_f(6751468363564694789),
            const_f(10581122285882655867),
        ]),
        u: QuinticExtension([
            const_f(16822879694511882841),
            const_f(7030889609682609080),
            const_f(1819733726510865699),
            const_f(1477354361991598818),
            const_f(3060932650955723086),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(16383078186728412911),
            const_f(17336793750234608284),
            const_f(10282704501742138249),
            const_f(8902952211247569575),
            const_f(10036728575538225007),
        ]),
        u: QuinticExtension([
            const_f(980771758638014650),
            const_f(8822864673362619613),
            const_f(1247272673889574430),
            const_f(8049338215992656959),
            const_f(5754772454101411592),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(4793164719180728081),
            const_f(10337661541467018847),
            const_f(4370608981419008671),
            const_f(8309057178611279515),
            const_f(11967697131554357119),
        ]),
        u: QuinticExtension([
            const_f(17586180332786867000),
            const_f(10992062529780955862),
            const_f(4283639578773926288),
            const_f(10598406479331979533),
            const_f(13292632801372322468),
        ]),
    },
];
pub(crate) const MUL_TABLE_G80: [AffinePoint; 16] = [
    AffinePoint {
        x: QuinticExtension([
            const_f(13832685079504880268),
            const_f(18013036221761440296),
            const_f(1301626881083565265),
            const_f(9139126253053898429),
            const_f(4505395467569954655),
        ]),
        u: QuinticExtension([
            const_f(7359813255592029850),
            const_f(16688014242518042008),
            const_f(4399996806448279465),
            const_f(5271684552135959425),
            const_f(11652444551874101645),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(10957597387983347508),
            const_f(15279113224094632648),
            const_f(16636004563247846338),
            const_f(139361871129849794),
            const_f(14913244377905888101),
        ]),
        u: QuinticExtension([
            const_f(7004241227096627206),
            const_f(639096603853214644),
            const_f(17343971022152731708),
            const_f(11127082727624914758),
            const_f(6961420809959752544),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(1678194015101167575),
            const_f(10443033913340861968),
            const_f(15723961754453665965),
            const_f(466551946746500778),
            const_f(1384638131140679955),
        ]),
        u: QuinticExtension([
            const_f(7911659739613756657),
            const_f(9008449922226900897),
            const_f(8828649835406020350),
            const_f(12804093940915848836),
            const_f(5168873490743917498),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(9668801441369446334),
            const_f(1618542760290755427),
            const_f(15806911258918325259),
            const_f(14508945524557601221),
            const_f(8400708218360666510),
        ]),
        u: QuinticExtension([
            const_f(2070702333293922760),
            const_f(6249392735673775978),
            const_f(5221268220067076678),
            const_f(12830382095618421300),
            const_f(6798253292813277552),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(613541603487791685),
            const_f(13807376311954113152),
            const_f(4937154484322350324),
            const_f(3044864073363788260),
            const_f(10659806245468237672),
        ]),
        u: QuinticExtension([
            const_f(11268721606331277338),
            const_f(14114972563238185761),
            const_f(15134656524184558801),
            const_f(8109827563124888851),
            const_f(3238236749755375190),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(3653303296020985061),
            const_f(16968816356701165477),
            const_f(3537989784194419792),
            const_f(6048563117397703739),
            const_f(13275594789417281589),
        ]),
        u: QuinticExtension([
            const_f(15320572452406052803),
            const_f(423975947193335924),
            const_f(9786061404780445812),
            const_f(113935901661183202),
            const_f(17462508908451992614),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(5575647366699441601),
            const_f(2189227564735866743),
            const_f(1686208091749425593),
            const_f(6736750915939348632),
            const_f(17433930427527644213),
        ]),
        u: QuinticExtension([
            const_f(7057911563532867792),
            const_f(16566118262655927325),
            const_f(12888897205414551370),
            const_f(14415855073450397097),
            const_f(1147090766535755807),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(15658118616953616040),
            const_f(2263539327323250613),
            const_f(2715487874579250798),
            const_f(11933376952724039169),
            const_f(17769318666901826029),
        ]),
        u: QuinticExtension([
            const_f(16745623139313228390),
            const_f(9536464142142244411),
            const_f(12504946243788089281),
            const_f(704708129354743638),
            const_f(14573477780244357666),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(13701409545959547673),
            const_f(11537398060095127371),
            const_f(18304316093185449069),
            const_f(16990637176495122248),
            const_f(8300454239136955447),
        ]),
        u: QuinticExtension([
            const_f(12946536999123301864),
            const_f(16028271018248917226),
            const_f(14442669626987508876),
            const_f(8204605677104061293),
            const_f(13012677989830312429),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(8977822175893189913),
            const_f(8385291758088962932),
            const_f(6459781748990922334),
            const_f(10500670301259390474),
            const_f(8148745850566531944),
        ]),
        u: QuinticExtension([
            const_f(648416469448933683),
            const_f(2018140447090876597),
            const_f(11059355864713025945),
            const_f(17171402628974174968),
            const_f(720667133464111689),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(533755834279203303),
            const_f(9133223350344638107),
            const_f(6496913004501565984),
            const_f(5070553496917221248),
            const_f(10026395429516732342),
        ]),
        u: QuinticExtension([
            const_f(17311718290481148297),
            const_f(12616184711972987746),
            const_f(16195499951758316636),
            const_f(8118955923598298529),
            const_f(16774524951584936403),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(8977932331742105562),
            const_f(9135241432935976918),
            const_f(8762554005485625681),
            const_f(14767442741287060847),
            const_f(9223537459805575058),
        ]),
        u: QuinticExtension([
            const_f(15269989054854026299),
            const_f(11931086694777575213),
            const_f(1979657370606607924),
            const_f(10082554692183350114),
            const_f(4573690475951190900),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(12857842861752747447),
            const_f(11647180289644065286),
            const_f(17408779236478002670),
            const_f(5917000661832739376),
            const_f(1047056879360966448),
        ]),
        u: QuinticExtension([
            const_f(60118689797675542),
            const_f(1664328840457595492),
            const_f(7075936368160047305),
            const_f(13974115771952400562),
            const_f(11318108364890349009),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(17452797179483233405),
            const_f(6882955852043132316),
            const_f(9304840691925828603),
            const_f(981483665863638676),
            const_f(11024236439678964632),
        ]),
        u: QuinticExtension([
            const_f(2608844450889021414),
            const_f(2862891036050959369),
            const_f(9059816914007502053),
            const_f(16849128770451662626),
            const_f(54944805734402483),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(7304267395998600281),
            const_f(12651945702824162423),
            const_f(12034846251704181244),
            const_f(14535937891251268540),
            const_f(16446125823956689442),
        ]),
        u: QuinticExtension([
            const_f(14013745143822621484),
            const_f(13346293440957348839),
            const_f(14559163781616146382),
            const_f(10079303505894311335),
            const_f(13316971442260780794),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(2073268421803561746),
            const_f(13903239987062959402),
            const_f(6595765789009484358),
            const_f(12734389031510939331),
            const_f(14507055985845886345),
        ]),
        u: QuinticExtension([
            const_f(6178525556615612),
            const_f(5187104181066643307),
            const_f(2097004975629951488),
            const_f(3624702972881058018),
            const_f(15835733836057682299),
        ]),
    },
];
pub(crate) const MUL_TABLE_G120: [AffinePoint; 16] = [
    AffinePoint {
        x: QuinticExtension([
            const_f(9358418073545563325),
            const_f(6201803925005767184),
            const_f(17525836657555505989),
            const_f(18172103331346227979),
            const_f(11525670089424228174),
        ]),
        u: QuinticExtension([
            const_f(15389027580004038174),
            const_f(17425413276694524614),
            const_f(15639145503384753087),
            const_f(15041017306226520945),
            const_f(7937401073912193639),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(424871884768762681),
            const_f(13522556051462729987),
            const_f(12578037128032095483),
            const_f(15478027026291985081),
            const_f(3107357372380600388),
        ]),
        u: QuinticExtension([
            const_f(139609698330600720),
            const_f(13047471464877067976),
            const_f(14569000597615364817),
            const_f(2241769726453036433),
            const_f(15809930333584099827),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(14490559385422698658),
            const_f(9192132350820542857),
            const_f(5174647998374408373),
            const_f(14517418341859680382),
            const_f(17127851909541764338),
        ]),
        u: QuinticExtension([
            const_f(10617869578552630251),
            const_f(15452062022333822112),
            const_f(74217513813449143),
            const_f(7065334431037916517),
            const_f(1908363005628198785),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(14158363767375738516),
            const_f(10881488396819614845),
            const_f(9845083246403658682),
            const_f(308084846693439896),
            const_f(2258456665285229766),
        ]),
        u: QuinticExtension([
            const_f(10189353602169967163),
            const_f(3307134994579671177),
            const_f(15193472587506759411),
            const_f(1522949334698619656),
            const_f(10335076055833410122),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6988575191781662507),
            const_f(6763011815163702392),
            const_f(12700583067108401780),
            const_f(10889091046959437472),
            const_f(14563326032896307580),
        ]),
        u: QuinticExtension([
            const_f(12122806272622858917),
            const_f(17957572904440664730),
            const_f(1371105162549165938),
            const_f(7050159476133204977),
            const_f(14174648605675469597),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(11134953890003368271),
            const_f(1950173651962543315),
            const_f(15717245132850143332),
            const_f(3404453732698149788),
            const_f(5301672891646287185),
        ]),
        u: QuinticExtension([
            const_f(3625450390591129442),
            const_f(7246221686985732698),
            const_f(883169685721066424),
            const_f(4890159692945065594),
            const_f(5846189492174531971),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(17779324141575511119),
            const_f(2222820233157145199),
            const_f(14311299573357024128),
            const_f(6091166172492559467),
            const_f(13251122054512244755),
        ]),
        u: QuinticExtension([
            const_f(13595785608342218333),
            const_f(5346420442473779380),
            const_f(15973815498598602014),
            const_f(17570023165337986853),
            const_f(4489084688781803549),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(18066103166710948306),
            const_f(17952420946495149357),
            const_f(6895726862811180419),
            const_f(5250742026743142449),
            const_f(5546233908977317256),
        ]),
        u: QuinticExtension([
            const_f(13627730136315133390),
            const_f(16318021942381891511),
            const_f(17522263726824223313),
            const_f(2960524358953784315),
            const_f(9229420628457238614),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(1295063301573260180),
            const_f(2809464405706641890),
            const_f(16876063007570590190),
            const_f(414980406456848047),
            const_f(8882993381636093379),
        ]),
        u: QuinticExtension([
            const_f(14084704505090840803),
            const_f(1455438701125484684),
            const_f(7140138141300391159),
            const_f(3304135812365795152),
            const_f(2617025679312300128),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6275460045614886806),
            const_f(3390801146218874506),
            const_f(17247998212939720068),
            const_f(14133145208463656732),
            const_f(3920522032578446900),
        ]),
        u: QuinticExtension([
            const_f(7628461038336051188),
            const_f(5939897916270777659),
            const_f(9105159200762125376),
            const_f(13546478897675664577),
            const_f(10279072558522952380),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(11630988947772602329),
            const_f(12620300429100070711),
            const_f(79628975116792272),
            const_f(17920472109136769182),
            const_f(5826732348459131885),
        ]),
        u: QuinticExtension([
            const_f(2736111763898189506),
            const_f(14407691554344511345),
            const_f(10405697919259369402),
            const_f(2951539272691560626),
            const_f(17028604616981679777),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(9988698078321716091),
            const_f(429119229031363106),
            const_f(7711926677839955310),
            const_f(14843425123144395632),
            const_f(2253491748118774140),
        ]),
        u: QuinticExtension([
            const_f(17190043005790419516),
            const_f(13808981798094567902),
            const_f(4645442529701115361),
            const_f(10360499666917437943),
            const_f(13003321814463836887),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(5428331454841389691),
            const_f(5911352608778299689),
            const_f(12033745745356201095),
            const_f(14100994707656604830),
            const_f(2886042088926452362),
        ]),
        u: QuinticExtension([
            const_f(12925133128294153456),
            const_f(6458535650167456730),
            const_f(8582452901418814402),
            const_f(9403948375821725222),
            const_f(4166244923628463342),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(14049231651863941702),
            const_f(9994040187462027964),
            const_f(16602305579940231446),
            const_f(8805602289491330699),
            const_f(544940053745291275),
        ]),
        u: QuinticExtension([
            const_f(18184165264127619754),
            const_f(11557606822284913524),
            const_f(7784129138807937081),
            const_f(11583517824597488539),
            const_f(7002309200501552489),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(814125571699125593),
            const_f(113298508670324430),
            const_f(3553512439231149575),
            const_f(5722734149611317431),
            const_f(13535892466294020417),
        ]),
        u: QuinticExtension([
            const_f(10718151468633124775),
            const_f(1411760656056230045),
            const_f(2150017719245220876),
            const_f(14735221082549759933),
            const_f(13642901740019011009),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(16483184730500681046),
            const_f(9673757784055259057),
            const_f(7760528659260895061),
            const_f(14112860811008950766),
            const_f(14165735631683045125),
        ]),
        u: QuinticExtension([
            const_f(16200754974233622593),
            const_f(15775772353572942080),
            const_f(8728522175126988968),
            const_f(14337787208807512369),
            const_f(6870309312996910338),
        ]),
    },
];
pub(crate) const MUL_TABLE_G160: [AffinePoint; 16] = [
    AffinePoint {
        x: QuinticExtension([
            const_f(4048997798593065056),
            const_f(8401406543098379712),
            const_f(8471972887547353150),
            const_f(11271856534362959532),
            const_f(11485893719004138771),
        ]),
        u: QuinticExtension([
            const_f(9981895593163975663),
            const_f(16506992680199754648),
            const_f(9795990766132909080),
            const_f(14537323266760073360),
            const_f(16786980505293186490),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(16515152542557678971),
            const_f(2820879738576535933),
            const_f(14546871004256087775),
            const_f(8067774721434663075),
            const_f(5547758516300176370),
        ]),
        u: QuinticExtension([
            const_f(13156895577790221631),
            const_f(14079823781876329633),
            const_f(3663423310046916033),
            const_f(8256729270602146828),
            const_f(8025936200066564880),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(7199041597631769070),
            const_f(6507380560962078664),
            const_f(8741238648067440929),
            const_f(5032023372661133788),
            const_f(1471499738040488525),
        ]),
        u: QuinticExtension([
            const_f(16127942173059622373),
            const_f(17662578881118466367),
            const_f(5426223217353814653),
            const_f(12687076501536075723),
            const_f(11700332978843695966),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(15540762768637989229),
            const_f(5977836013802283595),
            const_f(17611433126093706192),
            const_f(1869784237181444322),
            const_f(262436190082189342),
        ]),
        u: QuinticExtension([
            const_f(16646868690306195484),
            const_f(2492778147148350975),
            const_f(12994887025011189709),
            const_f(18073347299788553346),
            const_f(16182392324261935778),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(14451630909414751763),
            const_f(2682316750096868275),
            const_f(15784810705353479256),
            const_f(9913396490753039039),
            const_f(17084522528101355432),
        ]),
        u: QuinticExtension([
            const_f(13015512373883463322),
            const_f(18140315257280584894),
            const_f(5133502996496697434),
            const_f(581463011694460141),
            const_f(10720904114857970130),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(1736453911989778095),
            const_f(2522761214792928606),
            const_f(8490724482968195082),
            const_f(3061517266849590914),
            const_f(7560708607478466898),
        ]),
        u: QuinticExtension([
            const_f(2559867271025702686),
            const_f(8279186716530148418),
            const_f(9394033500068495079),
            const_f(15391096564340037389),
            const_f(15441682874751040991),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(10406668092884681305),
            const_f(12237016771705070337),
            const_f(6310848257747044271),
            const_f(4113320295997237722),
            const_f(16814466981743832206),
        ]),
        u: QuinticExtension([
            const_f(3167328430596803860),
            const_f(3373273028644416665),
            const_f(13236218152769964416),
            const_f(15816058495228292851),
            const_f(8001858317475143616),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(18177821317840763398),
            const_f(7208454950216370358),
            const_f(14780008596150434388),
            const_f(1996209710201147400),
            const_f(15053807226826426393),
        ]),
        u: QuinticExtension([
            const_f(15715262971932204033),
            const_f(8714266598318325282),
            const_f(16219555901832677748),
            const_f(245656264630859564),
            const_f(4633621313248689546),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(1841517068532537972),
            const_f(18159427598377627852),
            const_f(7101751901448687743),
            const_f(17419951806778701769),
            const_f(15872135176100603181),
        ]),
        u: QuinticExtension([
            const_f(14938501378296161155),
            const_f(1475670735048314023),
            const_f(16050270748361767813),
            const_f(14927165212644255889),
            const_f(11393545566941110440),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(11016016673823483108),
            const_f(10346862381783788104),
            const_f(15413371668364281620),
            const_f(3789574685442821016),
            const_f(10327416280296530490),
        ]),
        u: QuinticExtension([
            const_f(1471176826026130963),
            const_f(4450232675785892534),
            const_f(1999057422912905727),
            const_f(1862118471196890026),
            const_f(9836667920542412877),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6731903398643108337),
            const_f(9730751811492293683),
            const_f(14448312389075081409),
            const_f(12232946634578520226),
            const_f(5149728844990350383),
        ]),
        u: QuinticExtension([
            const_f(13304407859937134355),
            const_f(9591204855047500826),
            const_f(7113854334829183285),
            const_f(11795800474402144172),
            const_f(15515700514512556333),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(16206498404205552972),
            const_f(7467982993043588737),
            const_f(954111054908556531),
            const_f(15259102337638072429),
            const_f(12916169332967261393),
        ]),
        u: QuinticExtension([
            const_f(10986259094443142549),
            const_f(918816446526617182),
            const_f(10678622673672003543),
            const_f(9174304313393317665),
            const_f(7047157651466091392),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(5524339637429426536),
            const_f(13012396000187524883),
            const_f(8701257881797351665),
            const_f(7601128411527015893),
            const_f(16817462731082877836),
        ]),
        u: QuinticExtension([
            const_f(17315109416612436252),
            const_f(8903947754371488039),
            const_f(11080795620793054950),
            const_f(12186542410997831530),
            const_f(10711958746278079839),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6264744119896948359),
            const_f(10601340541921400101),
            const_f(14657733949024428965),
            const_f(4449426502181859631),
            const_f(15315608631820517742),
        ]),
        u: QuinticExtension([
            const_f(2840761601268004671),
            const_f(5838696228978373234),
            const_f(8592255273635329784),
            const_f(11032000397652854764),
            const_f(10686619933707435695),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(18403739294830496585),
            const_f(5395321127948182079),
            const_f(4362930334215698085),
            const_f(3891000203325226477),
            const_f(7114568565526390560),
        ]),
        u: QuinticExtension([
            const_f(3975610669088804605),
            const_f(17233183788370721900),
            const_f(3024806945190232867),
            const_f(12024175965114418277),
            const_f(16612390798970961761),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(16856493815994067196),
            const_f(14785817845078013270),
            const_f(2731707303337213832),
            const_f(5700860421257465777),
            const_f(16512159307408461032),
        ]),
        u: QuinticExtension([
            const_f(15635761722509813626),
            const_f(9688917495669656811),
            const_f(13645873987197106712),
            const_f(11818012498378673433),
            const_f(5396192277875563403),
        ]),
    },
];
pub(crate) const MUL_TABLE_G200: [AffinePoint; 16] = [
    AffinePoint {
        x: QuinticExtension([
            const_f(16579624480310836700),
            const_f(4654009893788464381),
            const_f(1879948550255376688),
            const_f(7165778526988411257),
            const_f(10027404176497435516),
        ]),
        u: QuinticExtension([
            const_f(14672132787094789329),
            const_f(8134912716135063128),
            const_f(1681926888624011127),
            const_f(3090601642585073427),
            const_f(5082367180675620723),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(12935460095664666921),
            const_f(13688862829955708769),
            const_f(9016000768964455819),
            const_f(18074632734324577885),
            const_f(10067423399607639746),
        ]),
        u: QuinticExtension([
            const_f(1891495241569347963),
            const_f(9444474092325495302),
            const_f(10783243003245947999),
            const_f(15112298120192081012),
            const_f(8489851093422035711),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(3971319576308477519),
            const_f(11846884681248592704),
            const_f(7252181329490053993),
            const_f(2837725043423724233),
            const_f(554712018738433440),
        ]),
        u: QuinticExtension([
            const_f(2283716927456626486),
            const_f(4678849096118793201),
            const_f(7064207072633614681),
            const_f(1293928013652227803),
            const_f(4122458298059420843),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(17348512312529533060),
            const_f(8639109793604178953),
            const_f(15818212126524653050),
            const_f(17950462779586277033),
            const_f(15262813331997381824),
        ]),
        u: QuinticExtension([
            const_f(5107077498491622223),
            const_f(4004992081347212098),
            const_f(13489481871700798330),
            const_f(1439663511274537768),
            const_f(2916087242841422420),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6922005184759837266),
            const_f(4816940458101140735),
            const_f(14348527682434315092),
            const_f(16536303157525555702),
            const_f(8297623336197847715),
        ]),
        u: QuinticExtension([
            const_f(3550074682900750241),
            const_f(5900089539068431592),
            const_f(15881485277116367548),
            const_f(4494234475833006435),
            const_f(698663099382505402),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(16241061472728124620),
            const_f(6131873802153215223),
            const_f(16831672901554690917),
            const_f(17254464099918200920),
            const_f(11185076059758094886),
        ]),
        u: QuinticExtension([
            const_f(4298246016297961963),
            const_f(7189403662133590696),
            const_f(9418905817123278198),
            const_f(14531204622533844239),
            const_f(8541862381303549676),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(15447868881789362057),
            const_f(5508196216379753233),
            const_f(16485811425555940264),
            const_f(13222911319795183488),
            const_f(3159840448703036170),
        ]),
        u: QuinticExtension([
            const_f(5204165561238044016),
            const_f(13468232899848292870),
            const_f(17191293205041837891),
            const_f(18246478932776430841),
            const_f(15325962678017236259),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(10621030779181154903),
            const_f(17136235063952920846),
            const_f(9314218722796245020),
            const_f(9368391401101022200),
            const_f(8555825846071793318),
        ]),
        u: QuinticExtension([
            const_f(7664574944617879289),
            const_f(16231709863136830941),
            const_f(6590015398331881523),
            const_f(14621275666438800255),
            const_f(6788090867699016859),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(720425677629731910),
            const_f(558256443318656554),
            const_f(5541446382714187419),
            const_f(16510886437312776757),
            const_f(10390794562653129460),
        ]),
        u: QuinticExtension([
            const_f(13345062980903998097),
            const_f(13515598458298192134),
            const_f(6777126340206327673),
            const_f(14815170113495224049),
            const_f(1808065573577174046),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(8512864041521600225),
            const_f(13632121416118897300),
            const_f(15219847883797542830),
            const_f(6281672652132756722),
            const_f(12690075810246041331),
        ]),
        u: QuinticExtension([
            const_f(15240394429738581893),
            const_f(13526765963383505570),
            const_f(385005971031338975),
            const_f(17055827423572264183),
            const_f(9799789282342151082),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(16433318271062727305),
            const_f(1053347735296699652),
            const_f(5450980641165850759),
            const_f(4054013846223550412),
            const_f(2947006303368928546),
        ]),
        u: QuinticExtension([
            const_f(17928731966205396993),
            const_f(17380491217072802345),
            const_f(11249617314541463800),
            const_f(13746866206588898967),
            const_f(12571294391280109436),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(18390535763930288817),
            const_f(13956792141128102015),
            const_f(14894537812189538984),
            const_f(5333289079328326940),
            const_f(10076243009389690036),
        ]),
        u: QuinticExtension([
            const_f(8250298621563594676),
            const_f(5752869300366626776),
            const_f(3645497280270257308),
            const_f(12900372348275640100),
            const_f(16885169851778635393),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(1544673605232492218),
            const_f(661976301925838846),
            const_f(18424167365360574137),
            const_f(3246102959607711481),
            const_f(5211114398364569488),
        ]),
        u: QuinticExtension([
            const_f(4460870711092798561),
            const_f(3451028986412879783),
            const_f(17189436277480328087),
            const_f(16695916816719405476),
            const_f(712205578119358045),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6293897706165296494),
            const_f(1058852550215266328),
            const_f(5340568372786241917),
            const_f(5059226109913370799),
            const_f(3526604109990729657),
        ]),
        u: QuinticExtension([
            const_f(11157536197710362632),
            const_f(12986275077072906620),
            const_f(3545776948579292831),
            const_f(11785840473114906984),
            const_f(10099190834060857641),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(11775056308366361359),
            const_f(797408970642010187),
            const_f(11281697728680415953),
            const_f(15083731163311144943),
            const_f(2591402698173474283),
        ]),
        u: QuinticExtension([
            const_f(11008763107345506753),
            const_f(5488815957510229275),
            const_f(14952094509887379098),
            const_f(11189563823429936956),
            const_f(5358872537390699328),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(7693346203646808522),
            const_f(3196464325266151639),
            const_f(1785222888514983133),
            const_f(4961084796993397121),
            const_f(16651160545753804914),
        ]),
        u: QuinticExtension([
            const_f(12113910774037980879),
            const_f(7934748119329239619),
            const_f(14520318444063438710),
            const_f(1372113091606068548),
            const_f(11259415352488711270),
        ]),
    },
];
pub(crate) const MUL_TABLE_G240: [AffinePoint; 16] = [
    AffinePoint {
        x: QuinticExtension([
            const_f(12150973993870501418),
            const_f(4223924024756880744),
            const_f(12164602482423882598),
            const_f(10110827219574637558),
            const_f(7454721448521923322),
        ]),
        u: QuinticExtension([
            const_f(8223067178251187472),
            const_f(14791411048736217143),
            const_f(6548050514357003677),
            const_f(14101051606185056042),
            const_f(9723051335063761713),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(2309936888761803695),
            const_f(10528374492977918782),
            const_f(2909110930817727979),
            const_f(14140458781369438628),
            const_f(14608954252678341690),
        ]),
        u: QuinticExtension([
            const_f(13315057417082143829),
            const_f(2875970576192442492),
            const_f(10204753160271556880),
            const_f(2528165599636440836),
            const_f(15588626368559095887),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(10285645489096016828),
            const_f(6826665230734386908),
            const_f(3643430412114742539),
            const_f(3525069461824492670),
            const_f(9265259914130088255),
        ]),
        u: QuinticExtension([
            const_f(18429224257556970829),
            const_f(16335577406386351411),
            const_f(1444816108348712587),
            const_f(532410028340092104),
            const_f(16527851406835121471),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(5704484539924172300),
            const_f(7404153046704669007),
            const_f(10464550607197363861),
            const_f(6247473471694475226),
            const_f(5115259736150878416),
        ]),
        u: QuinticExtension([
            const_f(17579790539786983406),
            const_f(2637134544147945869),
            const_f(4816977865203371123),
            const_f(248089872468508433),
            const_f(4531777203898089043),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(4467537824888613905),
            const_f(558399825893683724),
            const_f(4701759200819649961),
            const_f(16655886253669319016),
            const_f(14976096788667951951),
        ]),
        u: QuinticExtension([
            const_f(4542754722443867895),
            const_f(17838455475085664297),
            const_f(3856006738985174470),
            const_f(3695500756395218282),
            const_f(10605666420204608788),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(9600870350287519277),
            const_f(2943222643218798300),
            const_f(7504369701746722369),
            const_f(3618345531898965921),
            const_f(7996994629944741723),
        ]),
        u: QuinticExtension([
            const_f(1279526320710392206),
            const_f(4757632433269403318),
            const_f(12420546729136568420),
            const_f(17056471951401952929),
            const_f(16063059997803195687),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(38797529778910718),
            const_f(5399283910211167400),
            const_f(14916560969855131779),
            const_f(682297961769392616),
            const_f(18182112167162978281),
        ]),
        u: QuinticExtension([
            const_f(4851072938181616220),
            const_f(1768095373600786914),
            const_f(16165351733290258071),
            const_f(16718741168144865753),
            const_f(3387411250674432260),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(2078420199699477832),
            const_f(8841464556389390192),
            const_f(11642290600245563238),
            const_f(3963148268515541490),
            const_f(12483849286604430921),
        ]),
        u: QuinticExtension([
            const_f(13785261307443076347),
            const_f(8468941646155066103),
            const_f(3174688882704239544),
            const_f(10801045692115252746),
            const_f(12996585990193072559),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(9988106987444016195),
            const_f(550486750301142863),
            const_f(16147691900152849957),
            const_f(11482331300775839937),
            const_f(18210139443246491531),
        ]),
        u: QuinticExtension([
            const_f(17280645777130171058),
            const_f(5143971509316066734),
            const_f(9444564929039929588),
            const_f(2353260944176421839),
            const_f(7465399806142043858),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6553362623203065460),
            const_f(17774755178623960848),
            const_f(8170189958890581873),
            const_f(16479723056180470829),
            const_f(9052786989344840129),
        ]),
        u: QuinticExtension([
            const_f(5051652642644768336),
            const_f(8142249998939619774),
            const_f(6620402268383223033),
            const_f(15441849186338064088),
            const_f(17835312998647746744),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(12874799067286536937),
            const_f(5111767032451732361),
            const_f(9488358619465533651),
            const_f(2298827191559954790),
            const_f(15515015915136216385),
        ]),
        u: QuinticExtension([
            const_f(9934501897778259341),
            const_f(4862857445330881324),
            const_f(7191492445992175174),
            const_f(12588576141673201363),
            const_f(16820074689985814838),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(5506619209793629300),
            const_f(7913282297782618980),
            const_f(17035818002685942678),
            const_f(12219904669712698298),
            const_f(4701091471601382843),
        ]),
        u: QuinticExtension([
            const_f(15628068501760254685),
            const_f(9969915731376118609),
            const_f(4006095342913065224),
            const_f(11418313546696146922),
            const_f(9535581122323707943),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(660873376897289838),
            const_f(2332132165476400730),
            const_f(4904481730668053625),
            const_f(17592889807182765803),
            const_f(1775714498923493702),
        ]),
        u: QuinticExtension([
            const_f(8278491921012401650),
            const_f(255948487882786297),
            const_f(18072518402211877989),
            const_f(5587324201809627359),
            const_f(7916932786454127987),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(14880598518989845991),
            const_f(207047178807534206),
            const_f(8949411863433895830),
            const_f(15987292979823109393),
            const_f(10450748763888590480),
        ]),
        u: QuinticExtension([
            const_f(10555084898033032496),
            const_f(11149020781750632904),
            const_f(12754167684588738056),
            const_f(6203699237453069783),
            const_f(8397897173241663238),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(9626581523142877484),
            const_f(1014282284030781401),
            const_f(12559760948477539740),
            const_f(1719475860010180104),
            const_f(12167893974497751844),
        ]),
        u: QuinticExtension([
            const_f(10039328052268687164),
            const_f(16635482332793119899),
            const_f(5022923182724434224),
            const_f(13591886545913812687),
            const_f(4895263026932926029),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(15155727891419849945),
            const_f(3461317531761180793),
            const_f(16979908481664688625),
            const_f(15684045230954038314),
            const_f(12877152996799011381),
        ]),
        u: QuinticExtension([
            const_f(10120254644770986491),
            const_f(12192410531100649784),
            const_f(10938806981692604655),
            const_f(12172717977579895996),
            const_f(4275232645621155364),
        ]),
    },
];
pub(crate) const MUL_TABLE_G280: [AffinePoint; 16] = [
    AffinePoint {
        x: QuinticExtension([
            const_f(16213503882573976174),
            const_f(17168097236575729409),
            const_f(5196518270315815888),
            const_f(11117797779066091728),
            const_f(8133486084301919302),
        ]),
        u: QuinticExtension([
            const_f(11377245759937335205),
            const_f(4469833894127669069),
            const_f(9013706759438268290),
            const_f(1420430480105358672),
            const_f(16254559763550257786),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6770577214447522289),
            const_f(17115295583004483100),
            const_f(6277430884428490111),
            const_f(9367148506334403125),
            const_f(6974282321669735675),
        ]),
        u: QuinticExtension([
            const_f(1707192586757379005),
            const_f(16379422115255719397),
            const_f(9601719515238438547),
            const_f(18382556734962308004),
            const_f(9816132397810204232),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(9294285004075832264),
            const_f(17130662520620891577),
            const_f(17784936778486874242),
            const_f(7903567741417559125),
            const_f(13438250367827046909),
        ]),
        u: QuinticExtension([
            const_f(14848083910737694210),
            const_f(3876659422633582058),
            const_f(17589777829179053297),
            const_f(13255998440838131261),
            const_f(16836576774480954338),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6484667483670905518),
            const_f(14676664710267712890),
            const_f(943902085975544717),
            const_f(9447580128743448969),
            const_f(16970743407772865788),
        ]),
        u: QuinticExtension([
            const_f(6954165327706188094),
            const_f(8649474865423322710),
            const_f(2874401123529251159),
            const_f(6791369587301962541),
            const_f(4682935506184263557),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(17851907028754343206),
            const_f(804547578244465260),
            const_f(9836036663990543574),
            const_f(2782503787318970554),
            const_f(11029394870653732940),
        ]),
        u: QuinticExtension([
            const_f(17369554270592567524),
            const_f(11522732789192066880),
            const_f(10532626382274872331),
            const_f(15084091109637533903),
            const_f(12335999220635744679),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(17927819688989708837),
            const_f(988383065726890993),
            const_f(17134368434216800793),
            const_f(3745722005614877274),
            const_f(12079981168859675058),
        ]),
        u: QuinticExtension([
            const_f(16726774574446090464),
            const_f(16696890676634414315),
            const_f(1768034342698142990),
            const_f(5182686366441226421),
            const_f(12905524404643926664),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(5238880119446176935),
            const_f(7632489756740258264),
            const_f(13186772342659187888),
            const_f(5373269322406587505),
            const_f(4770529397079489612),
        ]),
        u: QuinticExtension([
            const_f(906031890843250730),
            const_f(2524575321869066878),
            const_f(1749353240118753004),
            const_f(8401611932919350607),
            const_f(13809067453022178888),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(3463236991239759171),
            const_f(9418504154128111760),
            const_f(16646612147210445285),
            const_f(8048290712504722073),
            const_f(4003118648580238214),
        ]),
        u: QuinticExtension([
            const_f(565998296113403270),
            const_f(5639331094891259297),
            const_f(3505572540820256764),
            const_f(828191569017542887),
            const_f(2857618747433407780),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(10579753460617358122),
            const_f(1425995342210623751),
            const_f(4437515648943912607),
            const_f(9208066594954079254),
            const_f(8133603054721359271),
        ]),
        u: QuinticExtension([
            const_f(5608659290599426924),
            const_f(14668957524891276508),
            const_f(3526636595086144132),
            const_f(16193609694652077957),
            const_f(14814479961293040846),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(6518242324859070685),
            const_f(10266975352945110329),
            const_f(7108371607348108785),
            const_f(14640665666349949967),
            const_f(6626682978649692287),
        ]),
        u: QuinticExtension([
            const_f(14497090493935686223),
            const_f(2359285113881335421),
            const_f(8857893277062885351),
            const_f(17134727430531764861),
            const_f(2965117089847599750),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(3234104227668768457),
            const_f(3225149987207949460),
            const_f(7086354559915607869),
            const_f(12877313486906605274),
            const_f(11561630661277747287),
        ]),
        u: QuinticExtension([
            const_f(2820816680951282830),
            const_f(16276533358496276797),
            const_f(8457579751747007027),
            const_f(227223614176367695),
            const_f(9322033005858872072),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(9250294019343021345),
            const_f(1766812231170355307),
            const_f(13463388795950836322),
            const_f(13360210575495818786),
            const_f(8017191608807726449),
        ]),
        u: QuinticExtension([
            const_f(2255099676665350867),
            const_f(9521276765352132752),
            const_f(2720865867697005972),
            const_f(11524331848149102745),
            const_f(1853382383896073031),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(2978539147748114010),
            const_f(8802332783321234516),
            const_f(8411186022628666251),
            const_f(8965939348601447933),
            const_f(114587463394660272),
        ]),
        u: QuinticExtension([
            const_f(10264182682530549009),
            const_f(9417221851288332824),
            const_f(6749125336792502108),
            const_f(5965179916047598634),
            const_f(5374111552073601171),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(11268491403320804445),
            const_f(3223633055331054259),
            const_f(4794616428042030393),
            const_f(9408476598474014080),
            const_f(9670252162084106811),
        ]),
        u: QuinticExtension([
            const_f(513179742804655698),
            const_f(6696701674770108433),
            const_f(15069423489583433755),
            const_f(8503371514376466366),
            const_f(9365587857178664019),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(8819593048375407845),
            const_f(6714540590843281274),
            const_f(12089230224772664935),
            const_f(17126824728760033332),
            const_f(15606471085374753320),
        ]),
        u: QuinticExtension([
            const_f(18421631694011443474),
            const_f(14887724257413490347),
            const_f(16780540965430713114),
            const_f(10821826807107857648),
            const_f(6670989693469822701),
        ]),
    },
    AffinePoint {
        x: QuinticExtension([
            const_f(3738402919300376943),
            const_f(3839435231606872366),
            const_f(9950700849677152681),
            const_f(13599115104880397736),
            const_f(8170114904882828684),
        ]),
        u: QuinticExtension([
            const_f(17403178918799426162),
            const_f(12276995036775716338),
            const_f(135085417294263740),
            const_f(6813815177605214808),
            const_f(15873392253765702165),
        ]),
    },
];

#[cfg(test)]
mod tests {
    use num::{BigUint, FromPrimitive};
    use plonky2::field::types::{Field, PrimeField64};

    use super::*;
    use crate::eddsa::curve::curve::ECgFp5Point;
    use crate::eddsa::curve::scalar_field::ECgFp5Scalar;

    // For k = 40*j (j = 0 to 7), constant Gk[] is an array of 16 points in
    // affine coordinates, with Gk[i] = (i+1)*(2^k)*G for the conventional
    // generator point G.

    fn compute_table(j: u64) -> [AffinePoint; 16] {
        let k = 40 * j;
        let mut table = [ECgFp5Point::NEUTRAL; 16];

        for i in 0..16 {
            let s_biguint =
                (BigUint::from_u64(1).unwrap() << k) * BigUint::from_usize(i + 1).unwrap();
            let s = ECgFp5Scalar::from_noncanonical_biguint(s_biguint);
            table[i] = ECgFp5Point::GENERATOR * s;
        }

        let mut res = [AffinePoint::NEUTRAL; 16];
        res.copy_from_slice(&ECgFp5Point::batch_to_affine(&table));

        res
    }

    fn print_table(table: &[AffinePoint; 16], name: &str) {
        println!("pub(crate) const {}: [AffinePoint; 16] = [", name);

        for i in 0..table.len() {
            let x_limbs = table[i]
                .x
                .0
                .map(|x| format!("const_f({})", x.to_canonical_u64()))
                .join(", ");
            let u_limbs = table[i]
                .u
                .0
                .map(|u| format!("const_f({})", u.to_canonical_u64()))
                .join(", ");

            println!(
                "    AffinePoint {{ x: QuinticExtension([{}]), u: QuinticExtension([{}]) }},",
                x_limbs, u_limbs
            );
        }

        println!("];");
    }

    #[test]
    #[ignore]
    fn print_mul_table() {
        for j in 0..8 {
            let table = compute_table(j);
            print_table(&table, &format!("MUL_TABLE_G{}", j * 40));
        }

        panic!();
    }
}
