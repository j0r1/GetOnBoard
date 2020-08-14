function main()
{
    console.log("main");
    const mod = import("./mainmodule.js?" + Date.now());
}

console.log("Maincode.js");
main();
window.getOnBoardMain = main;
