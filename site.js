$(document).ready(function(){
    $("pre").each(function(){
      $(this).wrapInner("<span></span>");
      var contentwidth = $(this).contents().width();
	    var blockwidth = $(this).width();
    if(contentwidth > blockwidth) {
    $(this).hover(function() {
      $(this).animate({ width: "965px"}, 250);
      }, function() {
      $(this).animate({ width: blockwidth }, 250);
      });
    }
    });
});
